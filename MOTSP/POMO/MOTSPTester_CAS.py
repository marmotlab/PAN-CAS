import torch
from torch.optim import Adam as Optimizer

import os
from logging import getLogger

from MOTSPEnv import TSPEnv as Env
from MOTSPModel_CAS import TSPModel as Model
from MOTSPModel_CAS import _get_encoding

from MOTSProblemDef_hyper import augment_xy_data_by_8_fold_2obj

from einops import rearrange

from utils.utils import *


class TSPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint_motsp-{epoch}.pt'.format(**model_load)
        self.checkpoint = torch.load(checkpoint_fullname, map_location=device)
        # self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

    def run(self, shared_problem, pref):
        self.time_estimator.reset()

        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        aug_score_list = torch.zeros(size=(0, 2))
        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            aug_score = self._test_one_batch(shared_problem, pref, batch_size, episode)
            aug_score_list = torch.cat((aug_score_list, aug_score), dim=0)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            all_done = (episode == test_num_episode)
            if all_done:
                for j in range(episode):
                    self.logger.info("AUG_OBJ_1 SCORE: {:.4f}, AUG_OBJ_2 SCORE: {:.4f} ".format(aug_score_list[j, 0],
                                                                                            aug_score_list[j, 1]))

        return aug_score_list.cpu().numpy()

    def _test_one_batch(self, shared_problem, pref, batch_size, episode):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1
        num_aug = aug_factor
        _, problem_size, _ = shared_problem.size()

        self.env.pomo_size = problem_size + 1
        self.env.batch_size = batch_size
        instances = shared_problem[episode: episode + batch_size]
        preference = pref[episode: episode + batch_size]

        if aug_factor == 8:
            instances = augment_xy_data_by_8_fold_2obj(instances).reshape(aug_factor, batch_size, problem_size, -1)
            preference = preference.repeat(aug_factor, 1).reshape(aug_factor, batch_size, -1)
        aug_score_list = []
        for i in range(num_aug):
            aug_factor = 1
            self.env.instances = instances[i]
            self.env.preference = preference[i]

            pref_size, obj_size = pref.size()
            distance_matrix = torch.linalg.norm(pref[None, :, :].expand(pref_size, pref_size, obj_size) - \
                              pref[:, None, :].expand(pref_size, pref_size, obj_size), dim=-1)
            _, sort_seq = torch.sort(distance_matrix)
            neighbor_id = sort_seq[:, :5]
            self.env.BATCH_IDX = torch.arange(self.env.batch_size)[:, None].expand(self.env.batch_size, self.env.pomo_size)
            self.env.POMO_IDX = torch.arange(self.env.pomo_size)[None, :].expand(self.env.batch_size, self.env.pomo_size)

            with torch.no_grad():
                self.model.load_state_dict(self.checkpoint['model_state_dict'])
                reset_state, _, _ = self.env.reset()
                problems = reset_state.instances

                pref = reset_state.preference
                self.model.decoder.assign(pref)
                self.model.pre_forward(problems)

            self.model.requires_grad_(False)
            # self.model.decoder.hyper_network.requires_grad_(
            #     True)
            self.model.decoder.single_head_key.requires_grad_(
                True)
            optimizer = Optimizer([self.model.decoder.single_head_key], lr=self.tester_params['param_lr'],
                                  weight_decay=self.tester_params['weight_decay'])
            # optimizer = Optimizer(self.model.decoder.hyper_network.parameters(), lr=self.tester_params['param_lr'],
            #                       weight_decay=self.tester_params['weight_decay'])
            # optimizer = Optimizer([self.model.decoder.hyper_network.parameters(), self.model.decoder.Wq_first_0,
            #                        self.model.decoder.Wq_last_0, self.model.decoder.W_out_0,
            #                        self.model.decoder.hyper_Wq_first, self.model.decoder.hyper_Wq_last,
            #                        self.model.decoder.hyper_multi_head_combine], lr=self.tester_params['param_lr'],
            #                       weight_decay=self.tester_params['weight_decay'])
            incumbent_solutions = torch.zeros(batch_size, problem_size, dtype=torch.int)

            for iter in range(self.tester_params['max_iteration']):

                incumbent_solutions_expand = incumbent_solutions.repeat(aug_factor, 1)

                prob_list = torch.zeros(size=(self.env.batch_size, self.env.pomo_size, 0))
                reset_state, _, _ = self.env.reset()
                problems = reset_state.instances

                pref = reset_state.preference
                state, reward, done = self.env.pre_step()
                # self.model.decoder.assign(pref)
                # self.model.decoder.set_kv(self.model.encoded_nodes)

                step = 0
                solutions = []
                first_action = (torch.arange(self.env.pomo_size) % problem_size)[None, :].expand(self.env.batch_size,
                                                                                                 self.env.pomo_size).clone()
                if iter > 0:
                    first_action[:, -1] = incumbent_solutions_expand[:, step]

                encoded_first_node = _get_encoding(self.model.encoded_nodes, first_action)
                # shape: (batch, pomo, embedding)
                self.model.decoder.set_q1(encoded_first_node)

                state, reward, done = self.env.step(first_action)
                solutions.append(first_action.unsqueeze(2))
                step += 1

                while not done:
                    probs = self.model(state)
                    # shape: (batch, pomo)
                    action = probs.reshape(self.env.batch_size * self.env.pomo_size, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(self.env.batch_size, self.env.pomo_size)

                    if iter > 0:
                        action[:, -1] = incumbent_solutions_expand[:, step]
                    state, reward, done = self.env.step(action)
                    solutions.append(action.unsqueeze(2))
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, action] \
                        .reshape(self.env.batch_size, self.env.pomo_size)

                    prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
                    step += 1

                solutions = torch.cat(solutions, dim=2)
                z = torch.ones(reward.shape).to(reward.device) * 0.0
                new_pref = pref[:, None, :].expand_as(reward)
                # ======================================================
                # expand neighbor solutions and rewards
                neighbor_reward = (reward.reshape(aug_factor, pref_size, -1, obj_size))
                neighbor_reward = rearrange(neighbor_reward, 'a b p h -> b (a p) h')[neighbor_id]
                # neighbor_reward_2 = reward[None, :, :, :].expand(pref_size, pref_size, -1, -1)[torch.arange(pref_size)[:, None].expand(neighbor_id.size()), neighbor_id]
                neighbor_solutions = solutions.reshape(aug_factor, pref_size, self.env.pomo_size, -1)
                neighbor_solutions = rearrange(neighbor_solutions, 'a b p h -> b (a p) h')[neighbor_id].reshape(pref_size, -1, neighbor_solutions.size(-1))

                neighbor_reward = - neighbor_reward
                neighbor_pref = new_pref.reshape(aug_factor, pref_size, self.env.pomo_size, -1)
                neighbor_pref = rearrange(neighbor_pref, 'a b p h -> b (a p) h')
                neighbor_z = z.reshape(aug_factor, pref_size, self.env.pomo_size, -1)
                neighbor_z = rearrange(neighbor_z, 'a b p h -> b (a p) h')

                neighbor_tch_reward = neighbor_pref[:, None, :, :].expand(neighbor_reward.size()) \
                                      * (neighbor_reward - neighbor_z[:, None, :, :].expand(neighbor_reward.size()))
                neighbor_tch_reward, _ = neighbor_tch_reward.max(dim=-1)
                neighbor_tch_reward = - neighbor_tch_reward.reshape(pref_size, -1)
                _, arg_max = neighbor_tch_reward.max(dim=-1)
                arg_max = arg_max.reshape(pref_size, 1)
                best_neighbor_solutions = torch.gather(neighbor_solutions, 1,
                                                   arg_max.unsqueeze(2).expand(-1, -1, solutions.shape[-1])).squeeze(1)
                # =======================================================
                # reward was negative, here we set it to positive to calculate TCH
                reward = - reward
                tch_reward = new_pref * (reward - z)
                tch_reward, _ = tch_reward.max(dim=2)

                reward = - reward
                tch_reward = -tch_reward

                cal_reward = tch_reward[:, :-1]
                log_prob = prob_list.log().sum(dim=2)
                # shape = (batch, group)

                tch_advantage = cal_reward - cal_reward.mean(dim=1, keepdim=True)
                tch_loss = -tch_advantage * log_prob[:, :-1]  # Minus Sign
                # shape = (batch, group)
                loss_1 = tch_loss.mean()
                loss_2 = -log_prob[:, -1].mean()

                tch_reward = tch_reward.reshape(aug_factor, batch_size, self.env.pomo_size)
                tch_reward_aug = rearrange(tch_reward, 'c b h -> b (c h)')

                _, max_idx_aug = tch_reward_aug.max(dim=1)
                max_idx_aug = max_idx_aug.reshape(max_idx_aug.shape[0], 1)
                max_reward_obj1 = rearrange(reward[:, :, 0].reshape(aug_factor, batch_size, self.env.pomo_size),
                                            'c b h -> b (c h)').gather(1, max_idx_aug)
                max_reward_obj2 = rearrange(reward[:, :, 1].reshape(aug_factor, batch_size, self.env.pomo_size),
                                            'c b h -> b (c h)').gather(1, max_idx_aug)

                aug_score = []
                aug_score.append(-max_reward_obj1.float())
                aug_score.append(-max_reward_obj2.float())

                incumbent_solutions = best_neighbor_solutions

                # Step & Return
                ################################################
                loss = loss_1 + loss_2 * self.tester_params['param_lambda']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            aug_score_list.append(torch.stack(aug_score, 0).transpose(1, 0).squeeze(2).contiguous())

        final_reward = rearrange(torch.stack(aug_score_list, 0),  'c b h -> b c h')
        final_z = torch.ones(final_reward.shape).to(final_reward.device) * 0.0
        final_tch_reward = (final_reward - final_z) * pref[:, None, :].expand(-1, num_aug, -1)
        final_tch_reward, _ = final_tch_reward.max(dim=2)
        _, final_max_idx = final_tch_reward.min(dim=1)
        final_reward = final_reward[torch.arange(pref_size), final_max_idx]

        return final_reward


