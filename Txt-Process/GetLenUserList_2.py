# -*- coding: utf-8 -*-
import os,os.path as op
from tqdm import tqdm
import re,ujson
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

P_TRAIN_VISIT_DIR = '/home/andrew/Desktop/Kaggle/competitions/IKCEST BigData/2019bigdata/train_visit_all'
P_TRAIN_IMAGE_DIR = '/home/andrew/Desktop/Kaggle/competitions/IKCEST BigData/2019bigdata/Open-Source Models/2019BaiduXJTU/data/train_image_raw/004'
S_TRAIN_VISIT_DIR = '../../Raw Final Data/train_visit'
#S_TRAIN_IMAGE_DIR = '../data/train_image_raw'
S_TEST_VISIT_DIR  = '../../Raw Final Data/test_visit'
S_TEST_IMAGE_DIR  = '../data/test_image_raw'

PTRAIN_DIR = 'PTrain_User_Txts'
STRAIN_DIR = 'STrain_User_Txts'
STEST_DIR  = 'STest_User_Txts'
MERGE_DIR  = 'Merge_Research'
ANS_DIR    = 'Ans_Dicts'

#Max Categories A user ever Appeared in.
max_cates_covered = 4
#Min Txts A user ever Appeared in totally.
min_txts_appeared = 3

#PTrain
PTr_User_Txts_path    = op.join(PTRAIN_DIR,'PTr_User_Txts.json')
#PTr_User_Txts_path    = 'dic1.json'
PTr_User_Nums_path    = op.join(PTRAIN_DIR,'PTr_User_Nums.json')
#STrain
STr_User_Txts_path    = op.join(STRAIN_DIR,'STr_User_Txts.json')
#STr_User_Txts_path    = 'dic2.json'
STr_User_Nums_path    = op.join(STRAIN_DIR,'STr_User_Nums.json')
#Merged
Merged_User_Txts_path = op.join(MERGE_DIR,'Merged_User_Txts.json')
Merged_User_Nums_path = op.join(MERGE_DIR,'Merged_User_Nums.json')
Merged_User_Dis_path  = op.join(MERGE_DIR,'Merged_User_Dis_{}_{}.json'.format(max_cates_covered,min_txts_appeared))
#STest
STe_User_Txts_path    = op.join(STEST_DIR,'STe_User_Txts.json')

CLASSES = ['00{}'.format(i) for i in range(1,10)]
#LABELS = ['P','S']

def AppendDict(Dict,key,val):
    if key not in Dict.keys():
        Dict[key] = [val]
    else:
        Dict[key].append(val)

def GetCate(txt):
    return re.findall(r'_(\d+)',txt)[0]

def SplitUserNumsByLen(User_Nums,json_dir):
    print('Start Split User_Nums by Len...')
    Len_User_Nums_List = [{} for CLASS in CLASSES]
    User_List = list(User_Nums.keys())
    for user in tqdm(User_List):
        numlist = User_Nums[user]
        Len_User_Nums_List[9-numlist.count(0)-1][user] = numlist
        del User_Nums[user]
    del User_List
    print('Split Over.\n')
    for CLASS in CLASSES:
        index = int(CLASS[-1])-1
        json_path = op.join(json_dir,'Len_User_Nums_{}.json'.format(index+1))
        ujson.dump(Len_User_Nums_List[index],open(json_path,'w+'))
        print('Saved {} Users to {}.'.format(len(Len_User_Nums_List[index]),json_path))
    del Len_User_Nums_List
    print('Save Over.\n')

def WriteListToTxt(lines,txt_path):
    f = open(txt_path,'w+')
    for line in tqdm(lines):
        f.writelines(line+'\n')
    f.close()

'''
Step1: Split (Under MERGE_DIR, X=1~9)
Split User_Nums(Merged) by Len:
User_Txts ==> Len_User_Nums_X.json*9 (Saved as 'Len_User_Nums_X.json'*9 under MERGE_DIR)
'''
#Load Merged_User_Nums
#User_Nums = ujson.load(open(Merged_User_Nums_path)) if op.isfile(Merged_User_Nums_path) else {}
#print('Loaded {} Users from {}.'.format(len(User_Nums),Merged_User_Nums_path))

#Split User_Txts ==> Len_User_Txts_X.json*9
#SplitUserNumsByLen(User_Nums,MERGE_DIR)
#del User_Nums


'''
Step2: Get User_List from Merged_Len_User_Nums_2
'''
#Merged_Len_User_Nums_2_path = op.join(MERGE_DIR,'Len_User_Nums_2.json')
#Merged_Len_User_List_2_path = op.join(MERGE_DIR,'Merged_Len_User_List_2.txt')
#User_Nums_2 = ujson.load(open(Merged_Len_User_Nums_2_path))
#User_List_2 = list(User_Nums_2.keys())
#WriteListToTxt(User_List_2,Merged_Len_User_List_2_path)



