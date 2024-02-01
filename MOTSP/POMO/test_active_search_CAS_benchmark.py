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

from MOTSPTester_CAS import TSPTester as Tester

from generate_test_dataset import load_dataset
from krodata_large_size.convet_benchmark_dataloader import load_data
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

logger_params = {
    'log_file': {
        'desc': 'test__tsp_n50',
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

    tester = Tester(
                    env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    if problem_size == 20:
        ref = np.array([15, 15])  # 20
    elif problem_size == 50:
        ref = np.array([30, 30])  # 50
    elif problem_size == 100:
        ref = np.array([60, 60])  # 100
    elif problem_size == 150:
        ref = np.array([90, 90])
    elif problem_size == 200:
        ref = np.array([120, 120])
    elif problem_size == 250:
        ref = np.array([150, 150])
    elif problem_size == 300:
        ref = np.array([180, 180])
    else:
        print('Have yet define a reference point for this problem size!')

    pref_list = torch.zeros(size=(0, 2)).to(device)
    for i in range(n_sols):
        pref = torch.zeros(2).to(device)
        pref[0] = 1 - 0.01 * i
        pref[1] = 0.01 * i
        pref_list = torch.cat((pref_list, pref[None, :]), dim=0)

    shared_problem = load_data(problem_size)
    # aug_score_list = []
    for j in range(len(shared_problem)):
        start_time = time.time()
        problem = torch.FloatTensor(shared_problem[j]).expand(n_sols, problem_size, 4).to(device)
        aug_score = tester.run(problem, pref_list)

        np.save('PAN_kro_%d_%s.npy' % (problem_size, j), aug_score)
        hv = hvwfg.wfg(aug_score.astype(float), ref.astype(float))
        hv_ratio = hv / (ref[0] * ref[1])
        print(j, 'HV Ratio: {:.4f}'.format(hv_ratio))
        print(j, 'Time: {:.4f}'.format(time.time() - start_time))


##########################################################################################
if __name__ == "__main__":
    main()
