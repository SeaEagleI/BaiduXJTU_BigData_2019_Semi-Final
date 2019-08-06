# -*- coding: utf-8 -*-
import os,os.path as op
import numpy as np

train_txt = '../data/train/train.txt'
val_txt   = '../data/train/val.txt'
txt_tlpt  = './folds_split/{}_{}.txt'

def GetLines(txt_path):
    return [line for line in open(txt_path).readlines() if len(line)>0]

TrainList = GetLines(train_txt)
ValList   = GetLines(val_txt)
AllList   = TrainList+ValList

def WriteTxt(List,txt_path):
    f = open(txt_path,'w+')
    for line in List:
        f.writelines(line)
    if op.exists(txt_path):
        print('{}\t{}'.format(txt_path,len(List)))

def ReAllocation(AllList,fold=5):
    unit = int(len(AllList)/fold)
    for i in range(fold):
        j = fold-i-1
        TrainList = AllList[:unit*j]+AllList[unit*(j+1):]
        ValList   = AllList[unit*j:unit*(j+1)]
        WriteTxt(TrainList,txt_tlpt.format('train',i+1))
        WriteTxt(ValList,txt_tlpt.format('val',i+1))

#ReAllocation(AllList)
