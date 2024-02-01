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


from MOCVRPTester import CVRPTester as Tester
from MOCVRProblemDef import get_random_problems
from generate_test_dataset import load_dataset
##########################################################################################
import time
import hvwfg


from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.style.use('default')
##########################################################################################
# parameters
problem_size = 20
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
        'path': './result/saved_cvrp20_model',  # directory path of pre-trained model and log files saved.
        'epoch': 200, # epoch version of pre-trained model to laod.
    },
    'test_episodes': 100, 
    'test_batch_size': 100,
    'augmentation_enable': True,
    'aug_factor': 8,
    'aug_batch_size': 100 
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test__cvrp_n20',
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
def main(n_sols = 101):

    timer_start = time.time()
    logger_start = time.time()
    device = torch.device('cuda:0' if USE_CUDA is True else 'cpu')
    
    if DEBUG_MODE:
        _set_debug_mode()
    
    create_logger(**logger_params)
    _print_config()
    
    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)
    
    copy_all_src(tester.result_folder)
    
    sols = np.zeros([n_sols, 2])

    loaded_problem = load_dataset(
        '/home/qiang/Desktop/MOVP/PMOCO/test_data/movrp/movrp%d_test_seed1234.pkl' % (problem_size))
    shared_depot_xy, shared_node_xy, shared_node_demand, capacity = [], [], [], []
    for i in range(len(loaded_problem)):
        depot, loc, dem, cap = loaded_problem[i]
        shared_depot_xy.append(depot)
        shared_node_xy.append(loc)
        shared_node_demand.append(dem)
        capacity.append(cap)
    shared_depot_xy = torch.FloatTensor(shared_depot_xy).to(device)[:, None, :]
    shared_node_xy = torch.FloatTensor(shared_node_xy).to(device)
    shared_node_demand = torch.FloatTensor(np.array(shared_node_demand) / np.array(capacity)[:, None]).to(device)
    
    for i in range(n_sols):
        pref = torch.zeros(2).cuda()
        pref[0] = 1 - 0.01 * i
        pref[1] = 0.01 * i
        pref = pref / torch.sum(pref)
        pref = pref[None, :].expand(len(loaded_problem), 2)
    
        aug_score = tester.run(shared_depot_xy, shared_node_xy, shared_node_demand,pref)
        sols[i] = np.array(aug_score)
        
    timer_end = time.time()
    
    total_time = timer_end - timer_start

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
    
    hv = hvwfg.wfg(sols.astype(float), ref.astype(float))
    hv_ratio = hv / (ref[0] * ref[1])

    print('Run Time(s): {:.4f}'.format(total_time))
    print('HV Ratio: {:.4f}'.format(hv_ratio))

##########################################################################################
if __name__ == "__main__":
    main()