# -*- coding: utf-8 -*-
from CalcFeatures import GetFeatures
import os,os.path as op
from tqdm import tqdm
import numpy as np
import re

TRAIN_VISIT_TXT_DIR = '../../Raw Final Data/train_visit'
TEST_VISIT_TXT_DIR  = '../../Raw Final Data/test_visit'

train_txt = '../data/train/train.txt'
val_txt   = '../data/train/val.txt'
npy_path  = 'Rows.npy'

def GetLines(txt_path):
    return [re.findall(r'/(.*?).jpg',line)[0]+'.txt' for line in open(txt_path).read().split('\n') if len(line)>0]

def GenerateCSVs(src_dir,flist,start=0,unit=10000):
    Rows = []
    npy_path = 'Rows_{}.npy'.format(start+1)
    print('GenerateCSVs: [{}]'.format(src_dir))
    print('[{}]: {} --- {}'.format(start,unit*start,unit*(start+1)))
    for txt in tqdm(flist[unit*start:unit*(start+1)]):
        label = re.findall(r'_(.*?).txt',txt)[0] if '_' in txt else ''
        features = GetFeatures(op.join(src_dir,txt),label)
        Rows.append(list(features.values()))
    np.save(npy_path,np.array(Rows))
    if np.load(npy_path).shape[0]==min(len(flist),unit*(start+1))-unit*start:
        print('Over')

TrainList = GetLines(train_txt)
ValList   = GetLines(val_txt)
TestList  = sorted(os.listdir(TEST_VISIT_TXT_DIR))

#GenerateCSVs(TRAIN_VISIT_TXT_DIR,TrainList[10000:],7)
#GenerateCSVs(TRAIN_VISIT_TXT_DIR,ValList,1)
GenerateCSVs(TEST_VISIT_TXT_DIR,TestList,9)

