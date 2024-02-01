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

from MOTSPTester_CAS_3obj import TSPTester as Tester

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
        'path': './result/train__tsp_n100',  # directory path of pre-trained model and log files saved.
        'epoch': 200,  # epoch version of pre-trained model to laod.
    },
    'test_episodes': 105,
    'test_batch_size': 105,
    'augmentation_enable': True,
    'aug_factor': 8,
    'aug_batch_size': 105,
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


def das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta, depth):
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            das_dennis_recursion(ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1)


def das_dennis(n_partitions, n_dim):
    if n_partitions == 0:
        return np.full((1, n_dim), 1 / n_dim)
    else:
        ref_dirs = []
        ref_dir = np.full(n_dim, np.nan)
        das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return np.concatenate(ref_dirs, axis=0)

##########################################################################################
def main(n_sols=105):
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

    tester = Tester(
                    env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    sols = np.zeros([n_sols, 3])

    # if problem_size == 100:
    # tester_params['aug_batch_size'] = 1
    # if tester_params['augmentation_enable']:
    #     test_name = 'Aug'
    #     tester_params['test_batch_size'] = tester_params['aug_batch_size']
    # else:
    #     test_name = 'NoAug'
    if n_sols == 105:
        uniform_weights = torch.Tensor(das_dennis(13, 3))  # 105
    elif n_sols == 1035:
        uniform_weights = torch.Tensor(das_dennis(44, 3))  # 1035
    elif n_sols == 10011:
        uniform_weights = torch.Tensor(das_dennis(140, 3))  # 10011

    # torch.manual_seed(2)
    # shared_problem = torch.rand(size=(problem_size, 4))[None, :, :].expand(n_sols, problem_size, 4).to(device)
    loaded_problem = load_dataset(
            '/home/yangyibin/hyperPMOCO/test_data/motsp_3obj/motsp_3obj%d_new_test_data_seed1234.pkl' % (problem_size))[:20]
    shared_problem = torch.FloatTensor(loaded_problem).to(device)

    aug_score_list = []
    for i in range(shared_problem.size(0)):
        aug_score = tester.run(shared_problem[i][None, :, :].expand(uniform_weights.size(0), problem_size, -1), uniform_weights)
        aug_score_list.append(aug_score)

    timer_end = time.time()
    total_time = timer_end - timer_start
    print('Run Time(s): {:.4f}'.format(total_time))

    if problem_size == 20:
        ref = np.array([15, 15, 15])  # 20
    elif problem_size == 50:
        ref = np.array([30, 30, 30])  # 50
    elif problem_size == 100:
        ref = np.array([60, 60, 60])  # 100
    elif problem_size == 150:
        ref = np.array([90, 90, 90])
    elif problem_size == 200:
        ref = np.array([120, 120, 120])
    elif problem_size == 250:
        ref = np.array([150, 150, 150])
    elif problem_size == 300:
        ref = np.array([180, 180, 180])
    else:
        print('Have yet define a reference point for this problem size!')

    hv_list = []
    aug_score_list = np.array(aug_score_list)
    # np.save('CNH_EAS_N%d_%s.npy'%(problem_size, test_name), aug_score_list)
    # for j in range(shared_problem.size(0)):
    #     hv = hvwfg.wfg(aug_score_list[j].astype(float), ref.astype(float))
    #     hv_list.append(hv)
    # hv_mean = np.array(hv_list).mean()
    aug_score_list = aug_score_list.mean(0)
    hv_mean = hvwfg.wfg(aug_score_list.astype(float), ref.astype(float))
    hv_ratio = hv_mean / (ref[0] * ref[1] * ref[2])

    print('HV Ratio: {:.4f}'.format(hv_ratio))
    # plt.show()


##########################################################################################
if __name__ == "__main__":
    main()
