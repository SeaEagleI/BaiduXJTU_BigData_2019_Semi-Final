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
LABELS = ['P','S']

def AppendDict(Dict,key,val):
    if key not in Dict.keys():
        Dict[key] = [val]
    else:
        Dict[key].append(val)

def TxtlistToCateTxts(txtlist,label):
    dic = dict([[CLASS,[]] for CLASS in CLASSES])
    for txts in txtlist:
        dic[GetCate(txts[0])] = [label+txt.replace('.txt','') if txt[0] not in LABELS else txt.replace('.txt','') for txt in txts]
    return dic

def LabelTxtlist(txtlist,label):
    for i in range(len(txtlist)):
        txtlist[i] = [label+txt.replace('.txt','') if txt[0] not in LABELS else txt.replace('.txt','') for txt in txtlist[i]]
    return txtlist

def MergeTxtLists(txtlistp,txtlists):
    dicp,dics = TxtlistToCateTxts(txtlistp,'P'),TxtlistToCateTxts(txtlists,'S')
    txtlist = [dicp[CLASS]+dics[CLASS] for CLASS in CLASSES]
    return [txts for txts in txtlist if len(txts)>0]

def MergeDict(Dict,key,val):
    if key in Dict.keys():
        Dict[key] = MergeTxtLists(val,Dict[key])
    else:
        Dict[key] = LabelTxtlist(val,'P')

def GetCate(txt):
    return re.findall(r'_(\d+)',txt)[0]

def GetUserTxtsFromXJsons(user_txts_path,json_dir):
    User_Txts = {}
    if not op.isdir(json_dir):
        os.mkdir(json_dir)
    print('Generating User_Txts (all) from X.json...')
    for CLASS in CLASSES:
        json_path = op.join(json_dir,'{}.json'.format(CLASS[-1]))
        user_txts = ujson.load(open(json_path)) if op.isfile(json_path) else {}
        print('\nCLASS:\t{}\nLoaded {} users from {}'.format(CLASS,len(user_txts),json_path))
        print('UserList-Length: {}'.format(len(user_txts)))
        for user,txtlist in tqdm(user_txts.items()):
            AppendDict(User_Txts,user,txtlist)
        del user_txts
        print()
    ujson.dump(User_Txts,open(user_txts_path,'w+'))
    print('Total:\t{} Users'.format(len(User_Txts)))
    return User_Txts

def SplitUserTxtsByLen(User_Txts,json_dir):
    print('Start Split User_Txts by Len...')
    Len_User_Txts_List = [{} for CLASS in CLASSES]
    User_List = list(User_Txts.keys())
    for user in tqdm(User_List):
        txtlist = User_Txts[user]
        Len_User_Txts_List[len(txtlist)-1][user] = txtlist
        del User_Txts[user]
    del User_List
    print('Split Over.\n')
    for CLASS in CLASSES:
        index = int(CLASS[-1])-1
        json_path = op.join(json_dir,'Len_User_Txts_{}.json'.format(index+1))
        ujson.dump(Len_User_Txts_List[index],open(json_path,'w+'))
        print('Saved {} Users to {}.'.format(len(Len_User_Txts_List[index]),json_path))
    del Len_User_Txts_List
    print('Save Over.\n')

'''
Step1: Convertion (Under STRAIN_DIR, X=1~9)
X.json*9 (Raw User-Txtlist(1 dim) Dict in Cate 00X Before Merge)
===>
STr_User_Txts.json (Merge X.json*9 to STr_User_Txts.json(11.5G))
At Present, under STRAIN_DIR, Len_User_Nums_X.json*9 + STr_User_Nums.json(3.7G) had been Successfully generated from X.json*9 in some previous work)
'''
#STr_User_Txts = GetUserTxtsFromXJsons(STr_User_Txts_path,STRAIN_DIR)
STr_User_Txts = ujson.load(open(STr_User_Txts_path)) if op.isfile(STr_User_Txts_path) else {}
print('Loaded {} Users from {}.'.format(len(STr_User_Txts),STr_User_Txts_path))
PTr_User_Txts = ujson.load(open(PTr_User_Txts_path)) if op.isfile(PTr_User_Txts_path) else {}
print('Loaded {} Users from {}.'.format(len(PTr_User_Txts),PTr_User_Txts_path))

'''
Step2: Merge & Split (Under MERGE_DIR, X=1~9)
Merge PTr_User_Txts(1.6G) + STr_User_Txts(9.6G) ==> User_Txts (Saved as 'Merged_User_Txts.json'),
then Split User_Txts(Merged) by Len:
User_Txts ==> Len_User_Txts_X.json*9 (Saved as 'Len_User_Txts_X.json'*9 under MERGE_DIR)
'''
#Merge PTr_User_Txts(1.6G) + STr_User_Txts(9.6G) ==> User_Txts
User_Txts = {}
STr_User_List = list(STr_User_Txts.keys())
for user in tqdm(STr_User_List):
    User_Txts[user] = LabelTxtlist(STr_User_Txts[user],'S')
    del STr_User_Txts[user]
del STr_User_List
PTr_User_List = list(PTr_User_Txts.keys())
for user in tqdm(PTr_User_List):
    MergeDict(User_Txts,user,PTr_User_Txts[user])
    del PTr_User_Txts[user]
del PTr_User_List

del PTr_User_Txts
del STr_User_Txts
print('After Merge: {} Users in total.'.format(len(User_Txts)))

#Save User_Txts as Merged_User_Txts_path
ujson.dump(User_Txts,open(Merged_User_Txts_path,'w+'))
print('Merged User_Txts Saved to {}.\n'.format(len(User_Txts),Merged_User_Txts_path))

#Split User_Txts ==> Len_User_Txts_X.json*9
SplitUserTxtsByLen(User_Txts,MERGE_DIR)
del User_Txts


