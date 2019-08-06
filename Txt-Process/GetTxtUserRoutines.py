# -*- coding: utf-8 -*-
import os,os.path as op
from tqdm import tqdm
import re,ujson
import numpy as np
from random import shuffle

P_TRAIN_VISIT_DIR = '/home/andrew/Desktop/Kaggle/competitions/IKCEST BigData/2019bigdata/train_visit_all'
S_TRAIN_VISIT_DIR = '../../Raw Final Data/train_visit'
S_TEST_VISIT_DIR  = '../../Raw Final Data/test_visit'

PTRAIN_DIR      = 'PTrain_User_Txts'
STRAIN_DIR      = 'STrain_User_Txts'
STEST_DIR       = 'STest_User_Txts'
MERGE_DIR       = 'Merge_Research'
STORAGE_DIR     = 'User_Cate_Routs'
ANS_DIR         = 'Ans_Dicts'

MERGED_TRAIN_JSON_DIR = 'Merged_User_Txt_Jsons'
#MERGED_TEST_JSON_DIR  = 'Merged_User_Txt_Jsons'

CLASSES = ['00{}'.format(i) for i in range(1,10)]
LABELS  = ['P','S','']
CLASSES  = ['00{}'.format(i) for i in range(1,10)]
LABELS   = ['P','S','']
PATHS    = {'P':P_TRAIN_VISIT_DIR,'S':S_TRAIN_VISIT_DIR}
INTERVAL = [i for i in range(2,4)]

def AppendDict(Dict,key,val):
    if key not in Dict.keys():
        Dict[key] = [val]
    else:
        Dict[key].append(val)

def MergeDict(Dict,key,val):
    if key not in Dict.keys():
        Dict[key] = val
    else:
        Dict[key] = sorted(list(set(Dict[key]+val)))

def Path(txt):
    return op.join(PATHS[txt[0]],'{}.txt'.format(txt[1:]))

def GetCate(txt):
    return re.findall(r'_(\d+)',txt)[0]

def MergeRouts(routs_list):
    Routs = {}
    for rout in routs_list:
        for date,hours in rout.items():
            MergeDict(Routs,date,hours)
    return dict(sorted([[date,OneHot(hours)] for date,hours in Routs.items()],key=lambda x:x[0]))

def OneHot(hours):
    Hours = [0 for i in range(24)]
    for hour in hours:
        Hours[hour] += 1
    return Hours

def CalcHours(Hours):
#    Hours = [OneHot(hours) for hours in hours_list]
    return np.sum(np.array(Hours),axis=0)

def SumHours(Routs):
    return np.sum(np.array(list(Routs.values())))
#    return sum([len(hours) for hours in Routs.values()])

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

def GetJsonsFromTxt(User_Cate_Routs,txt,txt_dir):
    cate = GetCate(txt)
    User_Dict = LoadDictFromTxt(op.join(txt_dir,txt))
    for user,routs_str in User_Dict.items():
        if user not in User_Cate_Routs.keys():
            User_Cate_Routs[user] = {}
        rout_dic = LoadJsonFromRout(routs_str)
        AppendDict(User_Cate_Routs[user],cate,rout_dic)

def TxtPro(txt_dir,label,i=0,unit=5000):
    User_Cate_Routs = {}
    txtlist = sorted(os.listdir(txt_dir))
    start,end = unit*i,unit*(i+1)
    print('[{}]\n[{}] {} --- {} of {}'.format(label,i,start,end,len(txtlist)))
    for txt in tqdm(txtlist[start:end]):
        GetJsonsFromTxt(User_Cate_Routs,txt,txt_dir)
    User_Cate_Routs_path = op.join(STORAGE_DIR,'User_Cate_Routs_{}_{}_{}.json'.format(label,start,end))
    ujson.dump(User_Cate_Routs,open(User_Cate_Routs_path,'w+'))
    print('Saved to {}\n'.format(User_Cate_Routs_path))

for i in range(40,60):
#    TxtPro(P_TRAIN_VISIT_DIR,'P',i)
    TxtPro(S_TRAIN_VISIT_DIR,'S',i)


#valid = True
#User_Routs = {}
##for user,txtlist in tqdm(User_Txts.items()):
#for user,txtlist in tqdm(ItemList[:200]):
#    User_Routs[user],Len_Hours = {},[]
#    for txts in txtlist:
#        cate = GetCate(txts[0])
#        routs_list = [LoadUserRoutFromTxt(user,Path(txt)) for txt in txts]
##        User_Routs[user][cate] = MergeRouts(routs_list)
#        Routs = MergeRouts(routs_list)
##        print(Routs)
#        if SumHours(Routs)<min_cate_hours:
#            valid = False
#            break
#        Len_Hours.append(SumHours(Routs))
#        User_Routs[user][cate] = DateAttrSplit(Routs)
#        User_Routs[user] = ReshapeDict(User_Routs[user])
