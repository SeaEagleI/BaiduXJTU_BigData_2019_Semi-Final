# -*- coding: utf-8 -*-
import os,os.path as op
from tqdm import tqdm
import re,ujson
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from datetime import datetime,timedelta

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

#Merged
Merged_User_Txts_path = op.join(MERGE_DIR,'Merged_User_Txts.json')
Len_User_Txts_2_path = op.join(MERGE_DIR,'Len_User_Txts_2.json')
Merged_User_Dis_path  = op.join(MERGE_DIR,'Merged_User_Dis_{}_{}.json'.format(max_cates_covered,min_txts_appeared))
#STest
STe_User_Txts_path    = op.join(STEST_DIR,'STe_User_Txts.json')

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
    return np.sum(np.array(list(Routs.values())))
#    return sum([len(hours) for hours in Routs.values()])

def Interval(date_str,start_dt=datetime(2018,10,1)):
    return (datetime.strptime(date_str,'%Y%m%d')-start_dt).days

def GetGraphFromRouts(Cate_Routs):
    Graph = np.zeros([182,24])
    for cate,routs in Cate_Routs.items():
        for date,hours in routs.items():
            for hour in hours:
                x = Interval(date)
                if Graph[x][hour] in [0,int(cate)] or (hour>=1 and Graph[x][hour-1]==Graph[x][hour]) or hour==0:
                    Graph[x][hour] = int(cate)
    return Graph


#User_Txts = ujson.load(open(Len_User_Txts_2_path))
#ItemList = list(User_Txts.items())

valid = True
User_Routs = {}
#for user,txtlist in tqdm(User_Txts.items()):
for user,txtlist in tqdm(ItemList[:200]):
    User_Routs[user],Len_Hours = {},[]
    for txts in txtlist:
        cate = GetCate(txts[0])
        routs_list = [LoadUserRoutFromTxt(user,Path(txt)) for txt in txts]
#        User_Routs[user][cate] = MergeRouts(routs_list)
        Routs = MergeRouts(routs_list)
#        print(Routs)
        if SumHours(Routs)<min_cate_hours:
            valid = False
            break
        Len_Hours.append(SumHours(Routs))
        User_Routs[user][cate] = DateAttrSplit(Routs)
    if sum(Len_Hours)>=100 and valid and max(Len_Hours)/min(Len_Hours)<max_mm_ratio:
        User_Routs[user] = ReshapeDict(User_Routs[user])
    else:
        del User_Routs[user]
        valid = True

for user,DACates in User_Routs.items():
    print('\n[{}] {}'.format(user,len(User_Txts[user])))
    for DA,CateHours in DACates.items():
#        dates,DAStats = CateHours[cate]
        print('\t{}: {}'.format(DA,len(CateHours)))
        for cate,Hours in CateHours.items():
            print('\t\t\t{}:\t{}'.format(cate,Hours))

#for user,cate_routs in User_Routs.items():
#    print('\n[{}] {}'.format(user,len(User_Txts[user])))
#    for cate,routs in cate_routs.items():
##        print('\t[{}] {}'.format(cate,len(cate_routs[cate])))
##        for date,hours in routs.items():
##            print('\t\t{}:\t{}'.format(date,OneHot(hours)))
#        dates,DAStats = cate_routs[cate]
#        print('\t{}: {}'.format(cate,dates))
#        for DA,Hours in DAStats.items():
#            print('\t\t{}:\t{}'.format(DA,Hours))

#valid = True
#User_Graphs = {}
#for user,txtlist in tqdm(ItemList[:100]):
#    User_Graphs[user],Len_Hours = {},[]
#    for txts in txtlist:
#        cate = GetCate(txts[0])
#        routs_list = [LoadUserRoutFromTxt(user,Path(txt)) for txt in txts]
#        Routs = MergeRouts(routs_list)
#        if SumHours(Routs)<min_cate_hours:
#            valid = False
#            break
#        Len_Hours.append(SumHours(Routs))
#        User_Graphs[user][cate] = Routs
#    if sum(Len_Hours)>=100 and valid and max(Len_Hours)/min(Len_Hours)<max_mm_ratio:
#        User_Graphs[user] = GetGraphFromRouts(User_Graphs[user])
#    else:
#        del User_Graphs[user]
#        valid = True
#
#for user,Graph in User_Graphs.items():
#    print('\n[{}] {}'.format(user,len(User_Txts[user])))
#    for i in range(len(Graph)):
#        date_str = datetime.strftime(datetime(2018,10,1)+timedelta(i),'%Y%m%d')
#        print('\t{:<15}\t{}\t{}'.format(DateAttr[date_str],date_str,Graph[i]))

