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

CLASSES = ['00{}'.format(i) for i in range(1,10)]

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

def GetTop1(catelist):
    dic = {}
    for i in list(set(catelist)):
        dic[i] = catelist.count(i)
    cate,num = max([[key,val] for key,val in dic.items()],key=lambda x:x[1])
    ratio = num/len(catelist)
    return len(catelist),cate,ratio

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
#Static_User_Nums = ujson.load(open(Static_User_Nums_path))

#Calculate Abs_Static_User_Nums According to Threshold_size (The Num of Txts A Static User Appeared in its Static(Only) Category.)
threshold_size = 5
#Abs_Static_User_Nums = dict([[user,numlist] for user,numlist in tqdm(Static_User_Nums.items()) if max(numlist)>=threshold_size])
#print('Ans Length: {}/{}\tRatio: {}'.format(len(Abs_Static_User_Nums),len(Static_User_Nums),len(Abs_Static_User_Nums)/len(Static_User_Nums)*100))

#Calc/Save User_Cate & Cate_User
#User_Cate,Cate_User = {},{}
#for user,numlist in tqdm(Abs_Static_User_Nums.items()):
#    User_Cate[user] = CLASSES[np.argmax(numlist)]
#User_Cate = dict(sorted([[user,cate] for user,cate in User_Cate.items()],key=lambda x:x[0]))
#for user,cate in tqdm(User_Cate.items()):
#    AppendDict(Cate_User,cate,user)
#print('{}\t{}'.format('Total',len(User_Cate)))
#print()
#Cate_User = dict(sorted([[cate,userlist] for cate,userlist in Cate_User.items()],key=lambda x:len(x[1]),reverse=True))
#for cate,userlist in Cate_User.items():
#    print('{}\t{}'.format(cate,len(userlist)))
#
#ujson.dump(User_Cate,open(op.join(MERGE_DIR,'Merged_User_Cate_{}.json'.format(threshold_size)),'w+'))
#ujson.dump(Cate_User,open(op.join(MERGE_DIR,'Merged_Cate_User_{}.json'.format(threshold_size)),'w+'))

#Load STe_User_Txts.
#STe_User_Txts = ujson.load(open(STe_User_Txts_path))
#STe_User_List  = sorted(list(STe_User_Txts.keys()))
#Load User_Cate.
#User_Cate = ujson.load(open(op.join(MERGE_DIR,'Merged_User_Cate_{}.json'.format(threshold_size))))
#User_Cate_List = sorted(list(User_Cate.keys()))
#Calc InterSectionList
#intersection_list = list(set(STe_User_List).intersection(set(User_Cate_List)))
#print('{}\t{}\n'.format('Intersection',len(intersection_list)))

#Generate Cate_Txts & Txt_Users.
#Cate_Txts,Txt_Users = {},{}
#for CLASS in CLASSES:
#    Cate_Txts[CLASS] = []
#for user in tqdm(intersection_list):
#    Cate_Txts[User_Cate[user]] += STe_User_Txts[user]
#    for txt in STe_User_Txts[user]:
#        AppendDict(Txt_Users,txt,user)

#Generate Ans_Dict
Ans_Dict = {}
_list = list(Txt_Users.items())
shuffle(_list)
#for txt,users in tqdm(Txt_Users.items()):
for txt,users in tqdm(_list):
    cates = sorted([User_Cate[user] for user in users])
#Criteria Control
    _len,cate,ratio = GetTop1(cates)
#    if (_len>=10 and ratio>=0.9) or (_len>=20 and ratio>=0.8):
#    if (_len>=10 and ratio>=0.8):
#    if (_len>=10 and ratio>=0.8) or (_len>=3 and ratio>=0.9):
#    if (_len>=10 and ratio>=0.75) or (_len>=3 and ratio>=0.9):
    if (_len>=10 and ratio>=0.75) or (_len>=2 and ratio>=0.9):
#        print('{}\t{}\t{}\t{}'.format(txt,_len,cate,ratio))
        Ans_Dict[txt.replace('.txt','')] = cate
print('{}'.format(len(Ans_Dict)))

#Save Ans_Dict
digits = [10,0.75,2,0.9]
ujson.dump(Ans_Dict,open('Ans_Merged-STe_Criteria_{}_{}.json'.format('_'.join([str(i) for i in digits]),threshold_size),'w+'))

#Statistics Overview on Ans_Dict:
Statistics(Ans_Dict)

