# -*- coding: utf-8 -*-
import os,os.path as op
import torch
import numpy as np 
import pandas as pd
from tqdm import tqdm

TENSOR_DIR = './folds_test'
CLASSES = ['00{}'.format(i) for i in range(1,10)]

def LoadNpz(npz,dir_path=TENSOR_DIR):
    return np.load(op.join(dir_path,npz))

Net1_f1    = LoadNpz("Net1_raw_fold1_Test.npz")
Net1_f2    = LoadNpz("Net1_raw_fold2_Test.npz")
Net1_f3    = LoadNpz("Net1_raw_fold3_Test.npz")
Net1_f4    = LoadNpz("Net1_raw_fold4_Test.npz")
Net1_f5    = LoadNpz("Net1_raw_fold5_Test.npz")

Net2_f1    = LoadNpz("Net2_1_fold1_Test.npz")

Net3_f1    = LoadNpz("Net3_w_fold1_Test.npz")
Net3_f3    = LoadNpz("Net3_w_fold3_Test.npz")
Net3_f4    = LoadNpz("Net3_w_fold4_Test.npz")
Net3_f5    = LoadNpz("Net3_w_fold5_Test.npz")

Net4_f1    = LoadNpz("Net4_TTA_fold1_Test.npz")
Net4_f4    = LoadNpz("Net4_TTA_fold4_Test.npz")
Net4_f5    = LoadNpz("Net4_TTA_fold5_Test.npz")

Net5_f1    = LoadNpz("Net5_HR_fold1_Test.npz")
Net5_f2    = LoadNpz("Net5_HR_fold2_Test.npz")
Net5_f3    = LoadNpz("Net5_HR_fold3_Test.npz")
Net5_f4    = LoadNpz("Net5_HR_fold4_Test.npz")
Net5_f5    = LoadNpz("Net5_HR_fold5_Test.npz")

Net6_f1    = LoadNpz("Net6_Features_fold1_Test.npz")

Net7_f1    = LoadNpz("Net7_MS_fold1_Test.npz")
Net7_f4    = LoadNpz("Net7_MS_fold4_Test.npz")
Net7_f5    = LoadNpz("Net7_MS_fold5_Test.npz")

Net8_f1    = LoadNpz("Net8_MS_cat_fold1_Test.npz")
Net8_f2    = LoadNpz("Net8_MS_cat_fold2_Test.npz")
Net8_f3    = LoadNpz("Net8_MS_cat_fold3_Test.npz")
Net8_f4    = LoadNpz("Net8_MS_cat_fold4_Test.npz")
Net8_f5    = LoadNpz("Net8_MS_cat_fold5_Test.npz")

#Net1_5_f1  = LoadNpz("Net1-5_fold1_Test.npz")

targets = Net1_f1['targets']
#weights = [0.712920,0.696030,0.702160,0.705060,0.707250,0.619030]
weights = [1 for i in range(100)]
#weights = [1,1,1,2,2,1,1,1,1,1]

def MergeSubmit(X_Multi,X_Nets,submit_name):
    filename = '../submission/{}.txt'.format(submit_name)
    f = open(filename,'w+')
    for i in tqdm(range(len(X_Multi))):
        X_Merge = list(X_Multi[i])
        for X_Net in X_Nets:
            X_Merge += list(X_Net[i])
    #     del X_Merge[2]
    #    print(np.array(X_Merge).shape)
        preds = torch.zeros(len(CLASSES))
        for j in range(len(X_Merge)):
            preds = preds+torch.nn.functional.normalize(torch.from_numpy(np.array([X_Merge[j]])))*weights[j]
        _, pred = preds.data.topk(1, 1, True, True)
    #    print('{}\t{}'.format(targets[i][:-4],CLASSES[pred[0][0]]))
        f.writelines('{}\t{}\n'.format(targets[i],CLASSES[pred[0][0]]))
    f.close()

def SingleSubmit(Test_Predicts,submit_name):
    filename = '../submission/{}.txt'.format(submit_name)
    f = open(filename,'w+')
    for i in tqdm(range(len(Test_Predicts))):
    #    print('{}\t{}'.format(targets[i][:-4],CLASSES[pred[0][0]]))
        f.writelines('{}\t{}\n'.format(targets[i],CLASSES[Test_Predicts[i]]))
    f.close()


#SingleSubmit(Net1['predicts'],'Net1_raw')
#SingleSubmit(Net2['predicts'],'Net2_1')
#SingleSubmit(Net3['predicts'],'Net3_w')
#SingleSubmit(Net4['predicts'],'Net4_TTA')
#SingleSubmit(Net5['predicts'],'Net5_HR')
#SingleSubmit(Net6['predicts'],'Net6_Features')
#SingleSubmit(Net1_5['predicts'],'Net1-5')
#MergeSubmit(Net1_5_f1['X'],[],'Net1-5_f1')
#MergeSubmit(Net1_5_f1['X'],[Net1_f2['X'],Net1_f3['X'],Net1_f4['X'],Net1_f5['X']],'Net1_(f1-f5)+Net(2-5)_f1')
#MergeSubmit(Net1_f1['X'],[Net1_f2['X'],Net1_f3['X'],Net1_f4['X'],Net1_f5['X']],'Net1_(f1-f5)')
#MergeSubmit(Net1_5_f1['X'],[Net6_f1['X'],Net1_f2['X'],Net1_f3['X'],Net1_f4['X'],Net1_f5['X']],'Net1_(f1-f5)+Net(2-6)_f1_adjusted')

X_Nets = [
          Net1_f1['X'],Net1_f2['X'],Net1_f3['X'],Net1_f4['X'],Net1_f5['X'],
          Net2_f1['X'],
          Net3_f1['X'],Net3_f3['X'],Net3_f4['X'],Net3_f5['X'],
          Net4_f1['X'],Net4_f4['X'],Net4_f5['X'],
          Net5_f1['X'],Net5_f2['X'],Net5_f3['X'],Net5_f4['X'],Net5_f5['X'],
          Net6_f1['X'],
          Net7_f1['X'],Net7_f4['X'],Net7_f5['X'],
          Net8_f1['X'],Net8_f2['X'],Net8_f3['X'],Net8_f4['X'],Net8_f5['X']
          ]
MergeSubmit(X_Nets[0],X_Nets[1:],'ENSEMBLE_Nets(27)')

