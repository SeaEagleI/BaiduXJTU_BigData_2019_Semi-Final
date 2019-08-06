# -*- coding: utf-8 -*-
import os,os.path as op
from tqdm import tqdm
import re,ujson
import numpy as np
from random import shuffle

P_TRAIN_VISIT_DIR = '/home/andrew/Desktop/Kaggle/competitions/IKCEST BigData/2019bigdata/train_visit_all'
#P_TRAIN_IMAGE_DIR = '/home/andrew/Desktop/Kaggle/competitions/IKCEST BigData/2019bigdata/Open-Source Models/2019BaiduXJTU/data/train_image_raw/004'
S_TRAIN_VISIT_DIR = '../../Raw Final Data/train_visit'
#S_TRAIN_IMAGE_DIR = '../data/train_image_raw'
S_TEST_VISIT_DIR  = '../../Raw Final Data/test_visit'
#S_TEST_IMAGE_DIR  = '../data/test_image_raw'
STEST_DIR = 'STest_User_Txts'
MERGE_DIR = 'Merge_Research'

#PTrain+STrain: User_Nums
PTr_Static_User_Nums_path = op.join(MERGE_DIR,'PTr_Static_User_Nums.json')
STr_Static_User_Nums_path = op.join(MERGE_DIR,'STr_Static_User_Nums.json')
#Merged:
Static_User_Nums_path     = op.join(MERGE_DIR,'Merged_Static_User_Nums.json')
Cutted_User_Nums_path     = op.join(MERGE_DIR,'Cutted_User_Nums (Len 2).json')
#STest: User_Txts
STe_User_Nums_path        = op.join(STEST_DIR,'STe_User_Txts.json')

CLASSES = ['00{}'.format(i) for i in range(1,10)]

def AppendDict(Dict,key,val):
    if key not in Dict.keys():
        Dict[key] = [val]
    else:
        Dict[key].append(val)

def MergeDict(Dict,key,val):
    if key in Dict.keys():
        numlist = list(map(lambda x,y:x+y,Dict[key],val))
        if 9-numlist.count(0)==1:
            Dict[key] = numlist
            return ''
        else:
            return numlist
    else:
        Dict[key] = val
        return ''

def CmpTxts(txt1,txt2,dir1=S_TRAIN_VISIT_DIR,dir2=S_TRAIN_VISIT_DIR):
    txt1 = open(op.join(dir1,txt1)).read()
    txt2 = open(op.join(dir2,txt2)).read()
    return txt1==txt2

def GetCate(txt):
    return re.findall(r'_(.*?).txt',txt)[0]

def GetEstimatedCates(txt_dir):
    AllTxtList = sorted(os.listdir(txt_dir))
    for CLASS in CLASSES:
        txtlist = sorted([txt for txt in AllTxtList if GetCate(txt)==CLASS])
        print('Cate: {}\tTxtList-Length: {}\tRatio: {:.3f}%\tTest-Estimated: {}'.format(CLASS,len(txtlist),len(txtlist)/len(AllTxtList)*100,len(txtlist)/4))

def LoadDictFromTxt(txt,txt_dir):
    txt_path = op.join(txt_dir,txt)
    user_lines = dict([line.split('\t') for line in open(txt_path).read().split('\n') if len(line)>0])
    user_txts = dict([[user,txt] for user in user_lines.keys()])
    return user_txts#,user_lines

def GetTop1(catelist):
    dic = {}
    for i in list(set(catelist)):
        dic[i] = catelist.count(i)
    cate,num = max([[key,val] for key,val in dic.items()],key=lambda x:x[1])
    ratio = num/len(catelist)
    return len(catelist),cate,ratio

#PTr+STr ==> Merged_Static_User_Nums
#PTr_Static_User_Nums = {'1':[0,0,0,0,0,0,0,23,0],'2':[0,0,0,0,0,12,0,0,0],'3':[0,0,0,20,0,0,0,0,0]}
PTr_Static_User_Nums = ujson.load(open(PTr_Static_User_Nums_path)) if op.isfile(PTr_Static_User_Nums_path) else {}
print('Loaded {} Users from {}.'.format(len(PTr_Static_User_Nums),PTr_Static_User_Nums_path))

#STr_Static_User_Nums = {'1':[0,0,0,0,0,0,0,13,0],'2':[0,0,0,9,0,0,0,0,0],'4':[0,0,0,0,0,0,0,5,0]}
STr_Static_User_Nums = ujson.load(open(STr_Static_User_Nums_path)) if op.isfile(STr_Static_User_Nums_path) else {}
print('Loaded {} Users from {}.'.format(len(STr_Static_User_Nums),STr_Static_User_Nums_path))

print()
Cut_User_Nums = {}
for user,numlist in tqdm(PTr_Static_User_Nums.items()):
    mergelist = MergeDict(STr_Static_User_Nums,user,numlist)
    if mergelist!='':
        Cut_User_Nums[user] = mergelist

print('\nCutted {} Non-Static Users.'.format(len(Cut_User_Nums)))
print('After Merge: {} Users in total.'.format(len(STr_Static_User_Nums)))

#Cutted User_Nums
ujson.dump(Cut_User_Nums,open(Cutted_User_Nums_path,'w+'))
print('\n{} Saved.'.format(Cutted_User_Nums_path))

Static_User_Nums = STr_Static_User_Nums
del PTr_Static_User_Nums
del STr_Static_User_Nums

#Merge Static User_Nums
ujson.dump(Static_User_Nums,open(Static_User_Nums_path,'w+'))
print('{} Saved.'.format(Static_User_Nums_path))

#Static_User_Nums = ujson.load(open(Merged_Path))

