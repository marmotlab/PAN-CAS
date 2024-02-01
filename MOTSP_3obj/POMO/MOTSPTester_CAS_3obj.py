import torch
from torch.optim import Adam as Optimizer

import os
from logging import getLogger

from MOTSPEnv_3obj import TSPEnv as Env
from MOTSPModel_CAS_3obj import TSPModel as Model
from MOTSPModel_CAS_3obj import _get_encoding

from MOTSProblemDef_3obj import augment_xy_data_by_n_fold_3obj

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

        aug_score_list = torch.zeros(size=(0, 3))
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
                    self.logger.info("AUG_OBJ_1 SCORE: {:.4f}, AUG_OBJ_2 SCORE: {:.4f}, AUG_OBJ_3 SCORE: {:.4f} ".format
                                     (aug_score_list[j, 0], aug_score_list[j, 1], aug_score_list[j, 2]))

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
            instances = augment_xy_data_by_n_fold_3obj(instances, aug_factor).reshape(aug_factor, batch_size, problem_size, -1)
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

            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            with torch.no_grad():
                reset_state, pref, _, _ = self.env.reset()
                self.model.decoder.assign(pref)
                self.model.pre_forward(reset_state.problems)

            self.model.requires_grad_(False)
            self.model.decoder.single_head_key.requires_grad_(
                True)
            optimizer = Optimizer([self.model.decoder.single_head_key], lr=self.tester_params['param_lr'],
                                  weight_decay=self.tester_params['weight_decay'])
            incumbent_solutions = torch.zeros(batch_size, problem_size, dtype=torch.int)

            for iter in range(self.tester_params['max_iteration']):

                incumbent_solutions_expand = incumbent_solutions.repeat(aug_factor, 1)

                prob_list = torch.zeros(size=(self.env.batch_size, self.env.pomo_size, 0))
                reset_state, pref, _, _ = self.env.reset()
                state, reward, done = self.env.pre_step()

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

                if self.env_params['problem_size'] == 20:
                    ref_values = 8
                if self.env_params['problem_size'] == 50:
                    ref_values = 20
                if self.env_params['problem_size'] == 100:
                    ref_values = 45
                if self.env_params['problem_size'] == 150:
                    ref_values = 60
                if self.env_params['problem_size'] == 200:
                    ref_values = 75

                z = torch.ones(reward.shape).cuda() * ref_values
                new_pref = pref[:, None, :].expand_as(reward)
                theta = 0.1

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

                neighbor_d1 = torch.matmul((neighbor_z[:, None, :, :].expand
                    (neighbor_reward.size()) - neighbor_reward)[:, :, :, None, :], (neighbor_pref[:, None, :, :].expand
                    (neighbor_reward.size()))[:, :, :, :, None]).squeeze() \
                     / torch.linalg.norm(neighbor_pref[:, None, :, :].expand(neighbor_reward.size()), dim=-1)
                neighbor_d1_variant = neighbor_pref[:, None, :, :].expand(neighbor_reward.size())\
                                      / torch.linalg.norm(neighbor_pref[:, None, :, :].expand(neighbor_reward.size()),
                                                          dim=-1)[:, :, :, None] * neighbor_d1[:, :, :, None]
                neighbor_d2 = torch.linalg.norm(neighbor_z[:, None, :, :].expand(neighbor_reward.size()) -
                                                neighbor_reward - neighbor_d1_variant, dim=-1)
                neighbor_pbi_reward = neighbor_d1 - theta * neighbor_d2
                neighbor_tch_reward = neighbor_pbi_reward.reshape(pref_size, -1)

                _, arg_max = neighbor_tch_reward.max(dim=-1)
                arg_max = arg_max.reshape(pref_size, 1)
                best_neighbor_solutions = torch.gather(neighbor_solutions, 1,
                                                   arg_max.unsqueeze(2).expand(-1, -1, solutions.shape[-1])).squeeze(1)
                # =======================================================
                # reward was negative, here we set it to positive to calculate TCH
                reward = - reward
                d1 = torch.matmul((z - reward)[:, :, None, :], new_pref[:, :, :, None]).squeeze() \
                     / torch.linalg.norm(new_pref, dim=-1)
                d1_variant = new_pref / torch.linalg.norm(new_pref, dim=-1)[:, :, None] * d1[:, :, None]
                d2 = torch.linalg.norm(z - reward - d1_variant, dim=-1)
                pbi_reward = d1 - theta * d2
                tch_reward = pbi_reward

                reward = - reward

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
                max_reward_obj3 = rearrange(reward[:, :, 2].reshape(aug_factor, batch_size, self.env.pomo_size),
                                            'c b h -> b (c h)').gather(1, max_idx_aug)

                aug_score = []
                aug_score.append(-max_reward_obj1.float())
                aug_score.append(-max_reward_obj2.float())
                aug_score.append(-max_reward_obj3.float())

                incumbent_solutions = best_neighbor_solutions

                # Step & Return
                ################################################
                loss = loss_1 + loss_2 * self.tester_params['param_lambda']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            aug_score_list.append(torch.stack(aug_score, 0).transpose(1, 0).squeeze(2).contiguous())

        final_reward = rearrange(torch.stack(aug_score_list, 0), 'c b h -> b c h')
        final_z = torch.ones(final_reward.shape).cuda() * ref_values
        final_pref = pref[:, None, :].expand_as(final_reward)
        theta = 0.1

        d1 = torch.matmul((final_z - final_reward)[:, :, None, :], final_pref[:, :, :, None]).squeeze() \
             / torch.linalg.norm(final_pref, dim=-1)
        d1_variant = final_pref / torch.linalg.norm(final_pref, dim=-1)[:, :, None] * d1[:, :, None]
        d2 = torch.linalg.norm(final_z - final_reward - d1_variant, dim=-1)
        pbi_reward = d1 - theta * d2
        final_tch_reward = pbi_reward

        _, final_max_idx = final_tch_reward.max(dim=1)
        final_reward = final_reward[torch.arange(pref_size), final_max_idx]

        return final_reward


