import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import os,os.path as op
from tqdm import tqdm
import numpy as np
import torch.multiprocessing as mp

TENSOR_DIR = '.'
CONCAT_DATA_PATH  = './Concat_Test.npz'
CLASSES = ['001', '002', '003', '004', '005', '006', '007', '008', '009']

def MergeNpzs(NETS_PATH):
#    if op.isfile(Data_Path):
#        return
    Npzs = [np.load(Net_Path) for Net_Path in NETS_PATH]
    Length = len(list(Npzs[0]['X']))
    X,targets,predicts = [],Npzs[0]['targets'],[]
    for i in tqdm(range(Length)):
        Tensors = [torch.from_numpy(Npz['X'][i]) for Npz in Npzs]
        Concat_Matrix = torch.cat(Tensors).cpu().detach().numpy()
        X.append(Concat_Matrix)
        preds = torch.zeros(len(CLASSES))
        for Tensor in Tensors:
            preds = preds+torch.nn.functional.normalize(Tensor)
        _, pred = preds.data.topk(1, 1, True, True)
        predicts.append(pred[0][0].cpu().detach().numpy())
#        print('{}\t{}'.format(targets[i],pred[0][0].cpu().detach().numpy()))
#        print('{}\t{}'.format(targets[i],CLASSES[int(pred[0][0].cpu().detach().numpy())]))
    np.savez(CONCAT_DATA_PATH, X=np.array(X), targets=np.array(targets), predicts=np.array(predicts))

Exclusions = ['Net1-5_fold1_Val.npz','Net2_1_fold1_Val.npz']
NETS_PATH = sorted([i for i in os.listdir('.') if 'Val' in i and i not in Exclusions])
print(NETS_PATH)
#MergeNpzs()
