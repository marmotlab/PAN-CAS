import numpy as np
import hvwfg
import torch
from torch.utils.data import Dataset
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_data(num_nodes):
    kroA_data = np.loadtxt('clusdata/ClusterA%d.tsp'%num_nodes, skiprows=6, usecols=(1, 2), delimiter=' ', dtype=float)
    kroB_data = np.loadtxt('clusdata/ClusterB%d.tsp'%num_nodes, skiprows=6, usecols=(1, 2), delimiter=' ', dtype=float)
    # kroC_data = np.loadtxt('krodata/kroC%d.tsp'%num_nodes, skiprows=6, usecols=(1, 2), delimiter=' ', dtype=float)
    # kroD_data = np.loadtxt('krodata/kroD%d.tsp'%num_nodes, skiprows=6, usecols=(1, 2), delimiter=' ', dtype=float)
    max_a = np.max(kroA_data)
    max_b = np.max(kroB_data)
    # max_c = np.max(kroC_data)
    # max_d = np.max(kroD_data)
    print('max kroA:', max_a)
    print('max kroB:', max_b)
    # print('max kroC:', max_c)
    # print('max kroD:', max_d)

    kroA_data = kroA_data / max_a
    kroB_data = kroB_data / max_b
    # kroC_data = kroC_data / max_c
    # kroD_data = kroD_data / max_d

    ins_AB = np.concatenate((kroA_data, kroB_data),axis=1)[None, :, :]
    # ins_AC = np.concatenate((kroA_data, kroC_data),axis=1)[None, :, :]
    # ins_AD = np.concatenate((kroA_data, kroD_data),axis=1)[None, :, :]
    # ins_BC = np.concatenate((kroB_data, kroC_data),axis=1)[None, :, :]
    # ins_BD = np.concatenate((kroB_data, kroD_data),axis=1)[None, :, :]
    # ins_CD = np.concatenate((kroC_data, kroD_data),axis=1)[None, :, :]

    dataset = [ins_AB]
    EPF1 = np.loadtxt('clusdata/EPF-Lust-8.txt')[:, 1:] / np.array([max_a, max_b])  # AB


    if num_nodes == 100:
        ref = np.array([60, 60])  # 100
    elif num_nodes == 150:
        ref = np.array([90, 90])
    elif num_nodes == 200:
        ref = np.array([120, 120])
    elif num_nodes == 250:
        ref = np.array([150, 150])
    elif num_nodes == 300:
        ref = np.array([180, 180])
    else:
        print('Have yet define a reference point for this problem size!')

    hv1 = hvwfg.wfg(EPF1.astype(float), ref.astype(float))

    hv_ratio1 = hv1 / (ref[0] * ref[1])
    print('hv1: {:.4f}'.format(hv_ratio1))


    return dataset

if __name__=="__main__":
    load_data(100)


# class Kro_dataset(Dataset):
#
#     def __init__(self, num_nodes):
#         super(Kro_dataset, self).__init__()
#
#         x1 = np.loadtxt('krodata/kroA%d.tsp'%num_nodes, skiprows=6, usecols=(1, 2), delimiter=' ', dtype=float)
#         x1 = x1 / (np.max(x1,0))
#         x2 = np.loadtxt('krodata/kroB%d.tsp'%num_nodes, skiprows=6, usecols=(1, 2), delimiter=' ', dtype=float)
#         x2 = x2 / (np.max(x2,0))
#         x = np.concatenate((x1, x2),axis=1)
#         x = x.T
#         x = x.reshape(1, 4, num_nodes)
#
#         self.dataset = torch.from_numpy(x).float()
#         self.dynamic = torch.zeros(1, 1, num_nodes)
#         self.num_nodes = num_nodes
#         self.size = 1
#
#
#     def __len__(self):
#         return self.size
#
#     def __getitem__(self, idx):
#         # (static, dynamic, start_loc)
#         return (self.dataset[idx], self.dynamic[idx], [])