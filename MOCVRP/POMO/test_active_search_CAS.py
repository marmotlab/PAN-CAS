##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = True
CUDA_DEVICE_NUM = 0

##########################################################################################
# Path Config
import os
import sys
import torch
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

##########################################################################################
# import
import logging
from utils.utils import create_logger, copy_all_src

from MOCVRPTester_CAS import CVRPTester as Tester

from generate_test_dataset import load_dataset
##########################################################################################
import time
import hvwfg

from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.style.use('default')
##########################################################################################
# parameters

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128 ** (1 / 2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './result/train_cvrp_n100',  # directory path of pre-trained model and log files saved.
        'epoch': 200,  # epoch version of pre-trained model to laod.
    },
    'test_episodes': 101,
    'test_batch_size': 101,
    'augmentation_enable': True,
    'aug_factor': 8,
    'aug_batch_size': 101,
    'param_lr': 0.0041,
    'param_lambda': 0.013,
    'weight_decay': 1e-6,
    'max_iteration': 200
}
# if tester_params['augmentation_enable']:
#     tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test__tsp_n20',
        'filename': 'run_log'
    }
}


##########################################################################################
def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################
def main(n_sols=101):
    timer_start = time.time()
    device = torch.device('cuda:0' if USE_CUDA is True else 'cpu')

    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()
    problem_size = 150

    env_params = {
        'problem_size': problem_size,
        'pomo_size': problem_size,
    }

    sols = np.zeros([n_sols, 2])

    if problem_size == 20:
        ref = np.array([30, 3])  # 20
    elif problem_size == 50:
        ref = np.array([50, 3])  # 50
    elif problem_size == 100:
        ref = np.array([80, 3])  # 100
    elif problem_size == 150:
        ref = np.array([120, 3])  # 150
    elif problem_size == 200:
        ref = np.array([180, 3])  # 200
    elif problem_size == 250:
        ref = np.array([220, 3])  # 250
    elif problem_size == 300:
        ref = np.array([250, 3])  # 300
    else:
        print('Have yet define a reference point for this problem size!')

    tester = Tester(
        env_params=env_params,
        model_params=model_params,
        tester_params=tester_params)

    pref_list = torch.zeros(size=(0, 2)).to(device)
    for i in range(n_sols):
        pref = torch.zeros(2).to(device)
        pref[0] = 1 - 0.01 * i
        pref[1] = 0.01 * i
        pref_list = torch.cat((pref_list, pref[None, :]), dim=0)

    # torch.manual_seed(2)
    # shared_problem = torch.rand(size=(problem_size, 4))[None, :, :].expand(n_sols, problem_size, 4).to(device)
    loaded_problem = load_dataset(
            '/home/yangyibin/hyperPMOCO/test_data/movrp/movrp%d_new_test_data_seed1234.pkl' % (problem_size))[:20]
    aug_score_list = []
    hv_list = []
    for i in range(len(loaded_problem)):

        depot, loc, dem, cap = loaded_problem[i]
        shared_depot_xy = (torch.FloatTensor(depot).to(device))[None, None, :].expand(pref_list.size(0), 1, -1)
        shared_node_xy = (torch.FloatTensor(loc).to(device))[None, :, :].expand(pref_list.size(0), problem_size, -1)
        shared_node_demand = (torch.FloatTensor(np.array(dem) / np.array(cap)).to(device))[None, :].expand(pref_list.size(0),  -1)

        aug_score = tester.run(shared_depot_xy, shared_node_xy, shared_node_demand, pref_list)
        # aug_score = np.array(get_non_dominated(aug_score, final_pop=True))
        # hv = hvwfg.wfg(aug_score.astype(float), ref.astype(float))
        # hv_list.append(hv)
        aug_score_list.append(aug_score)

    timer_end = time.time()
    total_time = timer_end - timer_start
    print('Run Time(s): {:.4f}'.format(total_time))

    for j in range(len(aug_score_list)):
        hv = hvwfg.wfg(aug_score_list[j].astype(float), ref.astype(float))
        hv_list.append(hv)
    hv_mean2 = np.array(hv_list).mean()
    hv_ratio2 = hv_mean2 / (ref[0] * ref[1])
    aug_score_list = np.array(aug_score_list)
    aug_score_list = aug_score_list.mean(0)
    hv_mean1 = hvwfg.wfg(aug_score_list.astype(float), ref.astype(float))
    hv_ratio1 = hv_mean1 / (ref[0] * ref[1])
    print('HV Ratio1: {:.4f}'.format(hv_ratio1))
    print('HV Ratio2: {:.4f}'.format(hv_ratio2))
    # plt.show()


##########################################################################################
if __name__ == "__main__":
    main()
