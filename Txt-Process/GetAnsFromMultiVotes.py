# -*- coding: utf-8 -*-
import os,os.path as op
from tqdm import tqdm
import re,ujson
import numpy as np
from random import shuffle

P_TRAIN_VISIT_DIR = '/home/andrew/Desktop/Kaggle/competitions/IKCEST BigData/2019bigdata/train_visit_all'
S_TRAIN_VISIT_DIR = '../../Raw Final Data/train_visit'
S_TEST_VISIT_DIR  = '../../Raw Final Data/test_visit'
STEST_DIR = 'STest_User_Txts'
MERGE_DIR = 'Merge_Research'

#PTrain+STrain: User_Nums
PTr_Static_User_Nums_path = op.join(MERGE_DIR,'PTr_Static_User_Nums.json')
STr_Static_User_Nums_path = op.join(MERGE_DIR,'STr_Static_User_Nums.json')
#Merged:
Static_User_Nums_path     = op.join(MERGE_DIR,'Merged_Static_User_Nums.json')
#STest: User_Txts
STe_User_Txts_path = op.join(STEST_DIR,'STe_User_Txts.json')

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

CLASSES = ['00{}'.format(i) for i in range(1,10)]

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

#Load Static Users_Nums
Static_User_Nums = ujson.load(open(Static_User_Nums_path))

#Calculate Abs_Static_User_Nums According to Threshold_size (The Num of Txts A Static User Appeared in its Static(Only) Category.)
threshold_size = 3
#Abs_Static_User_Nums = dict([[user,numlist] for user,numlist in tqdm(Static_User_Nums.items()) if sum(numlist)>threshold_size])
#print('Ans Length: {}/{}\tRatio: {}'.format(len(Abs_Static_User_Nums),len(Static_User_Nums),len(Abs_Static_User_Nums)/len(Static_User_Nums)*100))

#Calc/Save User_Cate & Cate_User
#User_Dis,Cate_User = {},{}
User_Dis = {}
for user,numlist in tqdm(Static_User_Nums.items()):
    if sum(numlist)>threshold_size:
        User_Dis[user] = np.array(numlist)/sum(numlist)#CLASSES[np.argmax()]
#User_Dis = dict(sorted([[user,cate] for user,cate in User_Dis.items()],key=lambda x:x[0]))
#for user,dis in tqdm(User_Dis.items()):
#    AppendDict(Cate_User,dis,user)
print('{}\t{}'.format('Total',len(User_Dis)))
print()
#Cate_User = dict(sorted([[cate,userlist] for cate,userlist in Cate_User.items()],key=lambda x:len(x[1]),reverse=True))
#for cate,userlist in Cate_User.items():
#    print('{}\t{}'.format(cate,len(userlist)))

ujson.dump(User_Dis,open(op.join(MERGE_DIR,'Merged_User_Dis_Num1_{}.json'.format(threshold_size)),'w+'))
#ujson.dump(Cate_User,open(op.join(MERGE_DIR,'Merged_Cate_User_Num1_{}.json'.format('No-Threshold')),'w+'))

#Load STe_User_Txts.
STe_User_Txts = ujson.load(open(STe_User_Txts_path))
STe_User_List  = sorted(list(STe_User_Txts.keys()))
#Load User_Cate.
#User_Cate = ujson.load(open(op.join(MERGE_DIR,'Merged_User_Cate_Num1_{}.json'.format('No-Threshold'))))
User_Dis_List = sorted(list(User_Dis.keys()))
#Calc InterSectionList
intersection_list = list(set(STe_User_List).intersection(set(User_Dis_List)))
print('{}\t{}\n'.format('Intersection',len(intersection_list)))

#Generate Cate_Txts & Txt_Users.
Cate_Txts,Txt_Users = {},{}
##for CLASS in CLASSES:
##    Cate_Txts[CLASS] = []
#for user in tqdm(intersection_list):
##    Cate_Txts[User_Cate[user]] += STe_User_Txts[user]
#    for txt in STe_User_Txts[user]:
#        AppendDict(Txt_Users,txt,user)

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
ujson.dump(Ans_Dict,open('Ans_MSMVC_{}_{}.json'.format('_'.join([str(i) for i in criteria]),threshold_size),'w+'))

#Statistics Overview on Ans_Dict:
Statistics(Ans_Dict)

#Submit
#Generate Submission Txt
Base_txt = '../submission/81.5320%.txt'
Submit(False,Base_txt,Ans_path)


