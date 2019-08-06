# -*- coding: utf-8 -*-
import numpy as np
import os,os.path as op
import ujson,re

S_TRAIN_VISIT_DIR = '../../Raw Final Data/train_visit'

#Ans_PTrain_STest_txt  = 'Ans_PTrain-STest.txt'
Base_txt  = '../submission/81.4850%.txt'
#CmpPS_txt = '../submission/Cmp_P-S.txt'
#Ans_STrain_STest_txt  = 'Ans_STrain-STest.txt'
#Additional_Dict_json  = '../Tensors/Additional_Dict.json'
Additional_Dict_path  = './Ans_Dicts/Ans_MSMVC_4_3_10_0.4_2_0.5.json'
#Base_txt     = '../submission/ENSEMBLE_Nets(27).txt'
#CmpPS_txt   = '../submission/Cmp_S-S.txt'
Ans_all_txt  = './Ans_Dicts/Ans_all.txt'

CLASSES = ['00{}'.format(i) for i in range(1,10)]

def FHead(fpath):
    filename = fpath.split('/')[-1]
    fhead = '.'.join(filename.split('.')[:-1])
    return fhead.replace('%','')

def LoadDictFromTxt(txt_path):
    ResDict = {}
    lines = [line for line in open(txt_path).read().split('\n') if len(line)>0]
    for line in lines:
        id,ans = line.split('\t')
        ResDict[id] = ans
    return ResDict

def WriteDictToTxt(txt_path,Dict):
    f = open(txt_path,'w+')
    for id,ans in Dict.items():
        f.writelines('{}\t{}\n'.format(id,ans))
    f.close()
    if op.isfile(txt_path):
        print('Finished: {} Lines'.format(len(Dict.keys())))

def DictCover(base_dict,cover_dict):
    modified = 0
    for id,ans in cover_dict.items():
        if base_dict[id] != ans:
            modified += 1
            base_dict[id] = ans
    print('Modified:\t{}/{}\tRatio:{:.3f}'.format(modified,len(base_dict),modified/len(base_dict)*100))
    return base_dict

def Submit(write=True,Base_txt=Base_txt,Additional_Dict_path=Additional_Dict_path):
    Adjusted_txt = '../submission/{}_{}.txt'.format(FHead(Base_txt),FHead(Additional_Dict_path))
    #Adjusted_txt = '../submission/{}_{}.txt'.format(FHead(Base_txt),FHead(Ans_all_txt))
    print('[{}]\n'.format(Adjusted_txt))
    Base_Dict       = LoadDictFromTxt(Base_txt)
    Additional_Dict = ujson.load(open(Additional_Dict_path))
    Ans_Dict        = LoadDictFromTxt(Ans_all_txt)
    DictCover(Base_Dict,Additional_Dict)
    DictCover(Base_Dict,Ans_Dict)
    if write:
        WriteDictToTxt(Adjusted_txt,Base_Dict)

def Statistics(Ans_Dict):
	Cate_ID = {}
	for id,cate in Ans_Dict.items():
		if cate in Cate_ID.keys():
			Cate_ID[cate].append(id)
		else:
			Cate_ID[cate] = [id]
	Cate_ID = dict(sorted([[cate,idlist] for cate,idlist in Cate_ID.items()],key=lambda x:len(x[1]),reverse=True))
	for cate,idlist in Cate_ID.items():
		print('{}\t{}'.format(cate,len(idlist)))

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

def GetCate(txt):
    return re.findall(r'_(.*?).txt',txt)[0]

def GetTop1(Dis_List):
    Sum_Dis = AvgDis(Dis_List)
    ratio = max(Sum_Dis)
    cate = CLASSES[np.argmax(Sum_Dis)]
    return len(Dis_List),cate,ratio

def GetEstimatedCates(txt_dir,printable=True):
    AllTxtList = sorted(os.listdir(txt_dir))
    CateEstimates = {}
    for CLASS in CLASSES:
        txtlist = sorted([txt for txt in AllTxtList if GetCate(txt)==CLASS])
        if printable:
            print('Cate: {}\tTxtList-Length: {}\tRatio: {:.3f}%\tTest-Estimated: {}'.format(CLASS,len(txtlist),len(txtlist)/len(AllTxtList)*100,len(txtlist)/4))
        CateEstimates[CLASS] = int(len(txtlist)/4)
    return CateEstimates

#Submit()
CateEstimates = GetEstimatedCates(S_TRAIN_VISIT_DIR,False)
