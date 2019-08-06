# -*- coding: utf-8 -*-
import os,os.path as op
from tqdm import tqdm
import re,ujson
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from datetime import datetime,timedelta
from SubmitFromDicts import Statistics,Submit,GetTop1

P_TRAIN_VISIT_DIR = '/home/andrew/Desktop/Kaggle/competitions/IKCEST BigData/2019bigdata/train_visit_all'
S_TRAIN_VISIT_DIR = '../../Raw Final Data/train_visit'
S_TEST_VISIT_DIR  = '../../Raw Final Data/test_visit'

PTRAIN_DIR = 'PTrain_User_Txts'
STRAIN_DIR = 'STrain_User_Txts'
STEST_DIR  = 'STest_User_Txts'
MERGE_DIR  = 'Merge_Research'
ANS_DIR    = 'Ans_Dicts'

#Max Categories A user ever Appeared in.
max_cates_covered = 4
#Min Txts A user ever Appeared in totally.
min_txts_appeared = 3
#Min Total Days A user visited in his least-visited category.
min_cate_days = 5
#Min Total Hours A user visited in his least-visited category.
min_cate_hours = 20
#MM_ratio
max_mm_ratio = 3
#Min Total Hours A user appeared.
min_hours_appeared = 30

#Merged
Merged_User_Txts_path = op.join(MERGE_DIR,'Merged_User_Txts.json')
Len_User_Txts_2_path = op.join(MERGE_DIR,'Len_User_Txts_2.json')
Merged_User_Dis_path = op.join(MERGE_DIR,'Merged_User_Dis_{}_{}.json'.format(max_cates_covered,min_txts_appeared))
#STest
STe_User_Txts_path     = op.join(STEST_DIR,'STe_User_Txts.json')
IS_List_path           = op.join(MERGE_DIR,'IS_STe-Len2.txt')
IS_User_Txts_Len2_path = op.join(MERGE_DIR,'IS_User_Txts_Len2.json')

#DateAttr_path = 'WorkdaySplit.json'
DateAttr_path = 'Date Attributes Criteria.json'
DateAttr      = ujson.load(open(DateAttr_path))

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

def LoadJsonFromReshapedDict(reshaped_json):
    lines = [line for line in open(reshaped_json).read().split('\n') if len(line)>0]
    jsons = [ujson.loads(line) for line in lines]
    user_txts = dict([[json['_id'],json['txtlist']] for json in jsons])
    return user_txts

def LoadUserRoutFromTxt(user,txt_path):
    routines = re.findall(r'{}\t(.*?)\n'.format(user),open(txt_path).read())[0]
    rout_dic = {}
    days = routines.split(',')
    for day in days:
        date,hours_str = day.split('&')
        hours = [int(hour) for hour in hours_str.split('|')]
        rout_dic[date] = hours
    return rout_dic

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

def DateAttrSplit(Routs):
    DAStats = {}
    for date,hours in Routs.items():
        AppendDict(DAStats,DateAttr[date],hours)
    for DA,Hours in DAStats.items():
        DAStats[DA] = CalcHours(Hours)
    return dict(sorted([[DA,Hours] for DA,Hours in DAStats.items()],key=lambda x:x[0]))

def ReshapeDict(CateDAs):
    DACates = {}
    for cate,DAStats in CateDAs.items():
        for DA,Hours in DAStats.items():
            if DA not in DACates.keys():
                DACates[DA] = {}
            DACates[DA][cate] = Hours
    return DACates

def SumHours(Routs):
    return int(np.sum(np.array(list(Routs.values()))))
#    return sum([len(hours) for hours in Routs.values()])

def HoursPro(User_Txts_List,i=0,unit=10000):
    User_Hours = {}
    start,end = unit*i,unit*(i+1)
    print('[{}] {} --- {} of {}'.format(i,start,end,len(User_Txts_List)))
    for user,txtlist in tqdm(User_Txts_List[start:end]):
        numlist = [0 for i in range(9)]
        for txts in txtlist:
            cate = GetCate(txts[0])
            routs_list = [LoadUserRoutFromTxt(user,Path(txt)) for txt in txts]
            numlist[int(cate)-1] = SumHours(MergeRouts(routs_list))
        User_Hours[user] = numlist
    User_Hours_path = op.join(MERGE_DIR,'User_Hours_Len2_IS_{}_{}.json'.format(start,end))
    ujson.dump(User_Hours,open(User_Hours_path,'w+'))
    print('Over')

def MergeUserHours(key_str='User_Hours_Len2_IS_',dir_path=MERGE_DIR):
    paths = sorted([op.join(dir_path,i) for i in os.listdir(dir_path) if key_str in i])
    User_Hours_List = []
    for json_path in paths:
        User_Hours_List += list(ujson.load(open(json_path)).items())
    return dict(User_Hours_List)

'''Get User_Hours of Train'''
#User_Txts = ujson.load(open(Len_User_Txts_2_path))
#ItemList = list(User_Txts.items())

#User_Txts = ujson.load(open(Len_User_Txts_2_path))
#print('Loaded {} Users from {}.'.format(len(User_Txts),Len_User_Txts_2_path))
#intersection_list = eval(open(IS_List_path).read())
#IS_User_Txts = dict([[user,User_Txts[user]] for user in tqdm(intersection_list)])
#ujson.dump(IS_User_Txts,open(IS_User_Txts_Len2_path,'w+'))
#print('Saved {} Users to {}.'.format(len(IS_User_Txts),IS_User_Txts_Len2_path))

#User_Hours_path = op.join(MERGE_DIR,'User_Hours_Len2_IS.json')
#ujson.dump(User_Hours,open(User_Hours_path,'w+'))

#IS_User_Txts_List = list(ujson.load(open(IS_User_Txts_Len2_path)).items())
#print('Loaded {} Users from {}.'.format(len(IS_User_Txts_List),IS_User_Txts_Len2_path))

#HoursPro(IS_User_Txts_List,4)

#User_Hours = MergeUserHours()

User_Dis = {}
for user,numlist in tqdm(User_Hours.items()):
#    if sum(numlist)>=min_hours_appeared:
    User_Dis[user] = np.array(numlist)/sum(numlist)#CLASSES[np.argmax()]

print('{}\t{}'.format('Total',len(User_Dis)))
print()

#ujson.dump(User_Dis,open(op.join(MERGE_DIR,'Merged_User_Dis_Num1_{}.json'.format(min_hours_appeared)),'w+'))
#ujson.dump(Cate_User,open(op.join(MERGE_DIR,'Merged_Cate_User_Num1_{}.json'.format('No-Threshold')),'w+'))

#Load STe_User_Txts.
#STe_User_Txts = ujson.load(open(STe_User_Txts_path))
#STe_User_List  = sorted(list(STe_User_Txts.keys()))

User_Dis_List = sorted(list(User_Dis.keys()))
#Calc InterSectionList
#intersection_list = list(set(STe_User_List).intersection(set(User_Dis_List)))
#print('{}\t{}\n'.format('Intersection',len(intersection_list)))

#Generate Cate_Txts & Txt_Users.
Txt_Users = {}
for user in tqdm(User_Dis_List):
    for txt in STe_User_Txts[user]:
        AppendDict(Txt_Users,txt,user)

#Generate Ans_Dict
Ans_Dict = {}
_list = list(Txt_Users.items())
shuffle(_list)
#for txt,users in tqdm(Txt_Users.items()):
for txt,users in tqdm(_list):
    Dis_List = [User_Dis[user] for user in users]
#Criteria Control
    _len,cate,ratio = GetTop1(Dis_List)
#    if (_len>=10 and ratio>=0.9) or (_len>=20 and ratio>=0.8):
#    if (_len>=10 and ratio>=0.8):
#    if (_len>=10 and ratio>=0.8) or (_len>=3 and ratio>=0.9):
#    if (_len>=10 and ratio>=0.75) or (_len>=3 and ratio>=0.9):
    if (_len>=10 and ratio>=0.75) or (_len>=2 and ratio>=0.9):
#        print('{}\t{}\t{}\t{}'.format(txt,_len,cate,ratio))
        Ans_Dict[txt.replace('.txt','')] = cate
print('{}'.format(len(Ans_Dict)))

#Save Ans_Dict
criteria = [min_hours_appeared,
            10,0.9,
            20,0.8,
            ]

Ans_path = './{}/Ans_AdvM2_{}.json'.format(ANS_DIR,'_'.join([str(i) for i in criteria]))
#Ans_path = './{}/Ans_AdvM2_{}.json'.format(ANS_DIR,'part_no-postthre')
print('[{}]\n'.format(Ans_path))
ujson.dump(Ans_Dict,open(Ans_path,'w+'))

#Statistics Overview on Ans_Dict:
Statistics(Ans_Dict)

#Submit
#Generate Submission Txt
Base_txt = '../submission/81.5320%.txt'
Submit(False,Base_txt,Ans_path)

