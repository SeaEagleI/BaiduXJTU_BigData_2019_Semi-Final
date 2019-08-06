# -*- coding: utf-8 -*-
import os,os.path as op
from tqdm import tqdm
import re,ujson
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from SubmitFromDicts import Submit

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
min_txts_appeared = 2

#PTrain
PTr_User_Nums_path    = op.join(PTRAIN_DIR,'PTr_User_Nums.json')
#STrain
STr_User_Nums_path    = op.join(STRAIN_DIR,'STr_User_Nums.json')
#Merged
Merged_User_Nums_path = op.join(MERGE_DIR,'Merged_User_Nums.json')
Merged_User_Dis_path  = op.join(MERGE_DIR,'Merged_User_Dis_{}_{}.json'.format(max_cates_covered,min_txts_appeared))
#STest
STe_User_Txts_path    = op.join(STEST_DIR,'STe_User_Txts.json')

CLASSES = ['00{}'.format(i) for i in range(1,10)]
#ZEROES  = [0 for i in range(9)]

def AvgDis(Dis_List):
    Sum = [0 for i in range(9)]
    for line in Dis_List:
        Sum = np.add(Sum,line)
    return Sum/sum(Sum)

def AppendDict(Dict,key,val):
    if key not in Dict.keys():
        Dict[key] = [val]
    else:
        Dict[key].append(val)

#def AppendDictIndex(Dict,key,index,val):
#    if key not in Dict.keys():
#        Dict[key] = [0 for i in range(9)]
#    Dict[key][index] = val

def MergeDict(Dict,key,val):
    if key in Dict.keys():
        Dict[key] = list(map(lambda x,y:x+y,Dict[key],val))
    else:
        Dict[key] = val

def GetCate(txt):
    return re.findall(r'_(.*?).txt',txt)[0]

#def GetUserNums(txt_dir,user_nums_path,json_dir,txt_type='train'):
#    User_Nums = {}
#    if not op.isdir(json_dir):
#        os.mkdir(json_dir)
#    if txt_type=='train':
#        print('Generating & Reshaping Dict: User-Txtlist...')
#        for CLASS in CLASSES:
#            json_path = op.join(json_dir,'{}.json'.format(CLASS[-1]))
#            user_txts = ujson.load(open(json_path)) if op.isfile(json_path) else {}
#            print('\nCLASS:\t{}\nLoaded {} users from {}'.format(CLASS,len(user_txts),json_path))
#            print('UserList-Length: {}'.format(len(user_txts)))
#            for user,txtlist in tqdm(user_txts.items()):
#                AppendDictIndex(User_Nums,user,int(CLASS)-1,len(txtlist))
#            del user_txts
#            print()
#    ujson.dump(User_Nums,open(user_nums_path,'w+'))
#    return User_Nums

def GetTop1(Dis_List):
    Sum_Dis = AvgDis(Dis_List)
    ratio = max(Sum_Dis)
    cate = CLASSES[np.argmax(Sum_Dis)]
    return len(Dis_List),cate,ratio

def Statistics(ID_Cate):
	Cate_ID = {}
	for id,cate in ID_Cate.items():
		if cate in Cate_ID.keys():
			Cate_ID[cate].append(id)
		else:
			Cate_ID[cate] = [id]
	Cate_ID = dict(sorted([[cate,idlist] for cate,idlist in Cate_ID.items()],key=lambda x:len(x[1]),reverse=True))
	for cate,idlist in Cate_ID.items():
		print('{}\t{}'.format(cate,len(idlist)))

#Len_User_Txts ==> Len_User_Nums
def TxtsToNums(Len_User_Txts_path,TRAIN_DIR):
    _len = int(re.findall(r'_(\d+).json',Len_User_Txts_path)[0])
    Len_User_Txts = ujson.load(open(Len_User_Txts_path))
    Key_List = list(Len_User_Txts.keys())
    Len_User_Nums = {}
    print('UserList-Length: {}'.format(len(Len_User_Txts)))
    for user in tqdm(Key_List):
        for txts in Len_User_Txts[user]:
            AppendDict(Len_User_Nums,user,int(GetCate(txts[0]))-1,len(txts))
#        if 9-Len_User_Nums[user].count(0)!=_len:
#            print('Error.')
#            return
        del Len_User_Txts[user]
    json_path = op.join(TRAIN_DIR,'Len_User_Nums_{}.json'.format(_len))
#    json_path = op.join(TRAIN_DIR,'PTr_User_Nums.json')
    print('Cate-Distribution: {}\tUserList-Length: {}'.format(_len,len(Len_User_Nums)))
#    print('Cate-Distribution: {}\tUserList-Length: {}'.format('all',len(Len_User_Nums)))
    ujson.dump(Len_User_Nums,open(json_path,'w+'))
    print('{} Saved.'.format(json_path))
    del Len_User_Txts
    del Len_User_Nums

'''
Real Run 
Step1: Convertion:  
Len_User_Txts_X.json ===> Len_User_Nums_X.json
(X=1~9+all, X means A user-numlist dict of users who Appears in X Cates.)
'''
#PTr_Txts_PathList = sorted([op.join(PTRAIN_DIR,f) for f in os.listdir(PTRAIN_DIR) if 'Len_User_Txts' in f])
#for i in range(len(PTr_Txts_PathList)):
#    TxtsToNums(PTr_Txts_PathList[i],PTRAIN_DIR)
#TxtsToNums(op.join(PTRAIN_DIR,'PTr_User_Txts.json'),PTRAIN_DIR)

'''
Step2: Merge PTr_User_Nums & STr_User_Nums above Threshold_size
'''
#Load PTr_User_Txts/PTr_User_Nums from PTrain_VISIT_DIR
#Load STr_User_Txts/STr_User_Nums from STrain_VISIT_DIR
#PTr_User_Nums = ujson.load(open(PTr_User_Nums_path)) if op.isfile(PTr_User_Nums_path) else {}
#print('Loaded {} Users in Total.'.format(len(PTr_User_Nums)))
#STr_User_Nums = ujson.load(open(STr_User_Nums_path)) if op.isfile(STr_User_Nums_path) else {}
#print('Loaded {} Users in Total.'.format(len(STr_User_Nums)))
##
#
#for user,numlist in tqdm(PTr_User_Nums.items()):
#    MergeDict(STr_User_Nums,user,numlist)
##
#print('After Merge: {} Users in total.'.format(len(STr_User_Nums)))
#User_Nums = STr_User_Nums
#del PTr_User_Nums
#del STr_User_Nums
#ujson.dump(User_Nums,open(Merged_User_Nums_path,'w+'))
#User_Nums = ujson.load(open(Merged_User_Nums_path))
#print('Loaded {} Users from {}.'.format(len(User_Nums),Merged_User_Nums_path))
#
'''
Step3: Calc User_Dis & InterSectionList between User_Nums and STe_User_Txts
'''
#
##Calc/Save User_Dis
#User_Dis = {}
#for user,numlist in tqdm(User_Nums.items()):
#    if sum(numlist)>min_txts_appeared and 9-numlist.count(0)<=max_cates_covered:
#        User_Dis[user] = np.array(numlist)/sum(numlist)
#print('{}:\t{}'.format('Total',len(User_Dis)))
#print()
#
#ujson.dump(User_Dis,open(Merged_User_Dis_path,'w+'))
#User_Dis = ujson.load(open(Merged_User_Dis_path))
###
####Load STe_User_Txts.
#STe_User_Txts = ujson.load(open(STe_User_Txts_path))
#STe_User_List = sorted(list(STe_User_Txts.keys()))
##Load User_Cate.
#User_Dis_List = sorted(list(User_Dis.keys()))
##Calc InterSectionList
#intersection_list = list(set(STe_User_List).intersection(set(User_Dis_List)))
#print('{}\t{}\n'.format('Intersection',len(intersection_list)))

'''
Step4: Calc, Save & Analyze Ans_Dict
'''

##Generate Cate_Txts & Txt_Users.
#Txt_Users = {}
#for user in tqdm(intersection_list):
#    for txt in STe_User_Txts[user]:
#        AppendDict(Txt_Users,txt,user)

##Generate Ans_Dict
#Ans_Dict = {}
#_list = list(Txt_Users.items())
#shuffle(_list)
#for txt,users in tqdm(Txt_Users.items()):
#    Dis_List = [User_Dis[user] for user in users]
#    _len,cate,ratio = GetTop1(Dis_List)
#    if (_len>=10 and ratio>=0.35) or (_len>=2 and ratio>=0.51):
##        print('{}\t{}\t{}\t{}'.format(txt,_len,cate,ratio))
#        Ans_Dict[txt.replace('.txt','')] = cate
#print('{}'.format(len(Ans_Dict)))
#
##Save Ans_Dict
#criteria = [max_cates_covered,min_txts_appeared,
#            10,0.35,
#            2,0.51,
#            ]
#
#Ans_path = './{}/Ans_MSMVC_{}.json'.format(ANS_DIR,'_'.join([str(i) for i in criteria]))
#print('[{}]\n'.format(Ans_path))
#ujson.dump(Ans_Dict,open(Ans_path,'w+'))

#Statistics Overview on Ans_Dict:
Statistics(Ans_Dict)

'''
Step5: Submit
'''
#Generate Submission Txt
Base_txt = '../submission/81.5320%.txt'
Submit(True,Base_txt,Ans_path)

