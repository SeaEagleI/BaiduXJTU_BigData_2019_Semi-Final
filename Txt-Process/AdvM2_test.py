# -*- coding: utf-8 -*-
import os,os.path as op
from tqdm import tqdm
import re,ujson
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from pymongo import MongoClient
from sklearn.preprocessing import normalize
from SubmitFromDicts import Submit

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
Len_User_Txts_3_path = op.join(MERGE_DIR,'Len_User_Txts_3.json')
Len_User_Txts_4_path = op.join(MERGE_DIR,'Len_User_Txts_4.json')
Merged_User_Dis_path  = op.join(MERGE_DIR,'Merged_User_Dis_{}_{}.json'.format(max_cates_covered,min_txts_appeared))
#STest
STe_User_Txts_path    = op.join(STEST_DIR,'STe_User_Txts.json')
#IS
IS_List_path       = op.join(MERGE_DIR,'IS_STe-Len2.txt')

CLASSES = ['00{}'.format(i) for i in range(1,10)]
#ZEROES  = [0 for i in range(9)]

#DateAttr_path = 'WorkdaySplit.json'
DateAttr_path = 'Date Attributes Criteria.json'
DateAttr      = ujson.load(open(DateAttr_path))

CLASSES  = ['00{}'.format(i) for i in range(1,10)]
LABELS   = ['P','S']
PATHS    = {'P':P_TRAIN_VISIT_DIR,'S':S_TRAIN_VISIT_DIR,'':S_TEST_VISIT_DIR}
INTERVAL = [i for i in range(2,4)]

def AppendDis(Dict,key,val):
    if key not in Dict.keys():
        Dict[key] = val
    else:
        Dict[key] += val

def MergeDict(Dict,key,val):
    if key not in Dict.keys():
        Dict[key] = val
    else:
        Dict[key] = sorted(list(set(Dict[key]+val)))

def Path(txt):
    return op.join(S_TEST_VISIT_DIR,txt)

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
        rout_dic[date] = OneHot(hours)
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
    return np.sum(np.array(Hours),axis=0).tolist()

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

def Stuff(Cates,Dis):
    ZEROS = np.zeros([9,])
    for i in range(len(Cates)):
        ZEROS[int(Cates[i])-1] = Dis[i]
    return ZEROS

def CalcDis(DAMetrics,DA_Hours):
    DA_ISList = list(set(list(DAMetrics)).intersection(set(list(DA_Hours))))
    SumDis = np.zeros([9,])
    for DA in DA_ISList:
        Cates,Credit,Weights = DAMetrics[DA]
        Hours = DA_Hours[DA]
        Dis = np.array(Weights).dot(Hours)#.dot(Credit)
        SumDis += Stuff(Cates,Dis)
    return SumDis

def GetTop1(Dis):
    ratio = max(Dis)/(max(Dis)+min(Dis))
    cate = CLASSES[np.argmax(Dis)]
    return len(Dis),cate,ratio

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

#User_Txts = ujson.load(open(Len_User_Txts_2_path))
#print('Loaded {} Users from {}.'.format(len(User_Txts),Len_User_Txts_2_path))
#STe_User_Txts = ujson.load(open(STe_User_Txts_path))
#print('Loaded {} Users from {}.'.format(len(STe_User_Txts),STe_User_Txts_path))
#
#STe_User_List = sorted(list(STe_User_Txts.keys()))
#Merged_User_List = sorted(list(User_Txts.keys()))
##Calc InterSectionList
#intersection_list = list(set(STe_User_List).intersection(set(Merged_User_List)))
#print('{}\t{}\n'.format('Intersection',len(intersection_list)))

#open(IS_List_path,'w+').write(str(intersection_list))
intersection_list = eval(open(IS_List_path).read())

#User_Dis_path = op.join(MERGE_DIR,'User_Hour_Dis_2_0.6.json')
#User_Dis = ujson.load(open(User_Dis_path))
#
#Txt_Dis = {}
#for user,DAMetrics in tqdm(User_Dis.items()):
#    txts = STe_User_Txts[user]
#    User_Dis_List = []
#    for txt in txts:
##        User_Dis = {'user':user,'txt':txt}
#        DA_Hours = DateAttrSplit(LoadUserRoutFromTxt(user,Path(txt)))
#        AppendDis(Txt_Dis,txt,CalcDis(DAMetrics,DA_Hours))
##        User_Dis_List.append(User_Dis)
##    print(User_Dis_List)
##    stest_txts_dis.insert_many(User_Dis_List)
#
###Generate Ans_Dict
#Ans_Dict = {}
#_list = list(Txt_Dis.items())
#shuffle(_list)
#for txt,dis in tqdm(_list):
#    _len,cate,ratio = GetTop1(dis)
##    if (_len>=10 and ratio>=0.4) or (_len>=2 and ratio>=0.51):
##        print('{}\t{}\t{}\t{}'.format(txt,_len,cate,ratio))
#    Ans_Dict[txt.replace('.txt','')] = cate
#print('{}'.format(len(Ans_Dict)))
#
##Save Ans_Dict
#criteria = [max_cates_covered,min_txts_appeared,
#            10,0.4,
#            2,0.51,
#            ]
#
##Ans_path = './{}/Ans_MSMVC_{}.json'.format(ANS_DIR,'_'.join([str(i) for i in criteria]))
#Ans_path = './{}/Ans_AdvM1_{}.json'.format(ANS_DIR,'part_no-postthre')
#print('[{}]\n'.format(Ans_path))
#ujson.dump(Ans_Dict,open(Ans_path,'w+'))

#Statistics Overview on Ans_Dict:
#Statistics(Ans_Dict)

'''
Step5: Submit
'''
#Generate Submission Txt
#Base_txt = '../submission/81.5320%.txt'
#Submit(Base_txt,Ans_path)

