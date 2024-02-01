##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
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

from MOTSPTester_3obj import TSPTester as Tester
from MOTSProblemDef_3obj import get_random_problems
from generate_test_dataset import load_dataset

##########################################################################################
import time
import hvwfg
import pickle

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.style.use('default')
##########################################################################################
# parameters
problem_size = 50
env_params = {
    'problem_size': problem_size,
    'pomo_size': problem_size,
}
model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
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
        'path': './result/20230712_075728_train__tsp_n50',  # directory path of pre-trained model and log files saved.
        'epoch': 200, 
    },
    'test_episodes': 100, 
    'test_batch_size': 100,
    'augmentation_enable': False,
    'aug_factor': 64,
    'aug_batch_size': 100,
    'n_sols': 105 #1035, 10011
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

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
timer_start = time.time()
logger_start = time.time()

def main(n_sols = tester_params['n_sols']):
    if DEBUG_MODE:
        _set_debug_mode()
    device = torch.device('cuda:0' if USE_CUDA is True else 'cpu')

    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    sols = np.zeros([n_sols, 3])

    loaded_problem = load_dataset(
        '/home/qiang/Desktop/MOVP/PMOCO/test_data/motsp_3obj/motsp_3obj%d_test_seed1234.pkl' % (problem_size))
    shared_problem = torch.FloatTensor(loaded_problem).to(device)

    if n_sols == 105:
        uniform_weights = torch.Tensor(das_dennis(13,3))  # 105
    elif n_sols == 1035:
        uniform_weights = torch.Tensor(das_dennis(44,3))   # 1035
    elif n_sols == 10011:
        uniform_weights = torch.Tensor(das_dennis(140,3))   # 10011
    
    for i in range(n_sols):
        pref = uniform_weights[i]
        pref = pref[None, :].expand(shared_problem.size(0), 3)
        aug_score = tester.run(shared_problem,pref)
        sols[i] = np.array(aug_score)
    
    timer_end = time.time()
    
    total_time = timer_end - timer_start

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
    
    hv = hvwfg.wfg(sols.astype(float), ref.astype(float))
    hv_ratio =  hv / (ref[0] * ref[1] * ref[2])
    
    print('Run Time(s): {:.4f}'.format(total_time))
    print('HV Ratio: {:.4f}'.format(hv_ratio))

##########################################################################################

if __name__ == "__main__":
    main()
