# -*- coding: utf-8 -*-
import os,os.path as op
from tqdm import tqdm
import re,ujson
import numpy as np
from pymongo import MongoClient
from random import shuffle

P_TRAIN_VISIT_DIR = '/home/andrew/Desktop/Kaggle/competitions/IKCEST BigData/2019bigdata/train_visit_all'
S_TRAIN_VISIT_DIR = '../../Raw Final Data/train_visit'
S_TEST_VISIT_DIR  = '../../Raw Final Data/test_visit'

PTRAIN_DIR      = 'PTrain_User_Txts'
STRAIN_DIR      = 'STrain_User_Txts'
STEST_DIR       = 'STest_User_Txts'
MERGE_DIR       = 'Merge_Research'
ANS_DIR         = 'Ans_Dicts'

#MERGED_TRAIN_JSON_DIR = 'Merged_User_Txt_Jsons'
#MERGED_TEST_JSON_DIR  = 'Merged_User_Txt_Jsons'

CLASSES = ['00{}'.format(i) for i in range(1,10)]
LABELS  = ['P','S','']

def LoadDictFromTxt(txt_path):
    ResDict = {}
    lines = [line for line in open(txt_path).read().split('\n') if len(line)>0]
    for line in lines:
        id,ans = line.split('\t')
        ResDict[id] = ans
    return ResDict

def LoadJsonFromRout(routines):
    rout_dic = {}
    days = routines.split(',')
    for day in days:
        date,hours_str = day.split('&')
        hours = [int(hour) for hour in hours_str.split('|')]
        rout_dic[date] = hours
    return rout_dic

def GetJsonsFromTxt(db,txt,txt_dir,label):
    User_Routs = LoadDictFromTxt(op.join(txt_dir,txt))
    for user,routs in User_Routs.items():
        rout_dic = LoadJsonFromRout(routs)
        line = {'txt':label+txt.replace('.txt',''),'user':user,'routs':rout_dic}
        db.insert_one(line)
    pass

def TxtPro(db,txt_dir,label,i=0,unit=40000):
    txtlist = sorted(os.listdir(txt_dir))
    start,end = unit*i,unit*(i+1)
    print('[{}]\n[{}] {} --- {} in {}'.format(label,i,start,end,txt_dir))
    for txt in tqdm(txtlist[start:end]):
        GetJsonsFromTxt(db,txt,txt_dir,label)


conn = MongoClient('localhost',27017)
merged_txts = conn.merged_txts
txt_routs = merged_txts.txt_routs

TxtPro(txt_routs,P_TRAIN_VISIT_DIR,'P',0)
#TxtPro(txt_routs,S_TRAIN_VISIT_DIR,'S',9)


