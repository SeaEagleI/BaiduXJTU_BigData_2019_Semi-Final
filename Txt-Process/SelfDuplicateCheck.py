# -*- coding: utf-8 -*-
import os,os.path as op
from tqdm import tqdm
import cv2
import re,json
import numpy as np
import matplotlib.pyplot as plt

P_TRAIN_VISIT_DIR = '/home/andrew/Desktop/Kaggle/competitions/IKCEST BigData/2019bigdata/train_visit_all'
P_TRAIN_IMAGE_DIR = '/home/andrew/Desktop/Kaggle/competitions/IKCEST BigData/2019bigdata/Open-Source Models/2019BaiduXJTU/data/train_image_raw/004'
S_TRAIN_VISIT_DIR = '../../Raw Final Data/train_visit'
S_TRAIN_IMAGE_DIR = '../data/train_image_raw'
S_TEST_VISIT_DIR  = '../../Raw Final Data/test_visit'
S_TEST_IMAGE_DIR  = '../data/test_image_raw'

Ans_PTrain_STest_txt  = 'Ans_PTrain-STest.txt'
Ans_STrain_STest_txt  = 'Ans_STrain-STest.txt'
Ans_STrain_STrain_txt = 'Ans_STrain-STrain.txt'

STrainSizeDict_path = 'STrainSizeDict.json'
IdenticalDict_PTrain_PTrain_path = 'IdenticalDict_PTr-PTr.json'
ClusterDict_PTrain_PTrain_path   = 'ClusterDict_PTr-PTr.json'

def AppendDict(Dict,key,val):
    if key in Dict.keys():
        Dict[key].append(val)
    else:
        Dict[key] = [val]

def RemoveSecondDim(List):
    ResList = []
    for line in tqdm(List):
        ResList += sorted(line)
    return ResList

def CmpTxts(txt1,txt2,dir1=S_TRAIN_VISIT_DIR,dir2=S_TRAIN_VISIT_DIR):
    txt1 = open(op.join(dir1,txt1)).read()
    txt2 = open(op.join(dir2,txt2)).read()
    return txt1==txt2

def GetSizeDict(dir_path):
    txtlist = sorted(os.listdir(dir_path))
    SizeDict = {}
    for txt in tqdm(txtlist):
        size = op.getsize(op.join(dir_path,txt))
        if size not in SizeDict.keys():
            SizeDict[size] = [txt]
        else:
            SizeDict[size].append(txt)
    SizeDict = dict(sorted(SizeDict.items(),key=lambda x:x[0]))
    return SizeDict

def WriteTxt(Dict,txt_path):
    f = open(txt_path,'w+')
    for stxt,ptxtlist in tqdm(Dict.items()):
        ans,id = re.findall(r'_(.*?).txt',ptxtlist[0])[0],re.findall(r'^(.*?).txt',stxt)[0]
        f.writelines('{}\t{}\n'.format(id,ans))
    f.close()
    if op.isfile(txt_path):
        print('Finished: {} Lines'.format(len(Dict.keys())))

def GetCate(txt):
    return re.findall(r'_(.*?).txt',txt)[0]

def Clustering(txtlist):
    txtlist = sorted(txtlist)
    Clusters = {}
    while len(txtlist)>0:
        ref_txt = txtlist[0]
        txtlist.remove(ref_txt)
        Clusters[ref_txt] = [ref_txt]
        copytxtlist = txtlist
        for txt in txtlist:
            if GetCate(txt)==GetCate(ref_txt) and CmpTxts(txt,ref_txt):
                Clusters[ref_txt].append(txt)
                copytxtlist.remove(txt)
        txtlist = copytxtlist
        Clusters[ref_txt] = sorted(Clusters[ref_txt])
    return list(Clusters.values())

def GetImgList(txtlist):
    return [txt.replace('.txt','.jpg') for txt in txtlist]

#S_Train--->S_Train
#STrainTxtList  = sorted(os.listdir(S_TRAIN_VISIT_DIR))
##STestTxtList   = sorted(os.listdir(S_TEST_VISIT_DIR))
#STrainSizeDict = json.load(open(STrainSizeDict_path)) if op.isfile(STrainSizeDict_path) else GetSizeDict(S_TRAIN_VISIT_DIR)
##STestSizeDict  = GetSizeDict(S_TEST_VISIT_DIR)
##OverLapList = [size for size in STrainSizeDict.keys() if size in STestSizeDict.keys()]
#OverLapList = sorted(list(STrainSizeDict.keys()))
#TxtIdenticalDict = {}
#
#for txt_size in tqdm(OverLapList):
#    txtlist = sorted(STrainSizeDict[txt_size])
#    Clusters = Clustering(txtlist)
#    TxtIdenticalDict[txt_size] = Clusters
##
##print(TxtIdenticalDict)
#
#TxtIdenticalDict = dict(sorted(TxtIdenticalDict.items(),key=lambda x:x[0]))
#Ratio = len(TxtIdenticalDict.keys())/len(STrainSizeDict)
#print('\n{}/{} {}%'.format(len(TxtIdenticalDict.keys()),len(STrainSizeDict),Ratio*100))
##WriteTxt(TxtIdenticalDict,Ans_STrain_STrain_txt)
#json.dump(TxtIdenticalDict,open(IdenticalDict_PTrain_PTrain_path,'w+'))

TxtIdenticalDict = json.load(open(IdenticalDict_PTrain_PTrain_path))
TxtIdenticalList = RemoveSecondDim(list(TxtIdenticalDict.values()))
ClusterDict = dict([[txtlist[0],GetImgList(txtlist)] for txtlist in TxtIdenticalList])
json.dump(ClusterDict,open(ClusterDict_PTrain_PTrain_path,'w+'))

OverTwoDict = dict([[txt,imglist] for txt,imglist in tqdm(ClusterDict.items()) if len(imglist)>=2])
Ratio = len(OverTwoDict)/len(ClusterDict)
print('\n{}/{}\t{}%'.format(len(OverTwoDict),len(ClusterDict),Ratio*100))
print('Duplicated Pics:\t{}'.format(sum([len(imglist) for imglist in OverTwoDict.values()])))
#for txt,imglist in list(ClusterDict.items())[:1000]:
#    if len(imglist)>1:
#        img_dir = op.join(S_TRAIN_IMAGE_DIR,GetCate(txt))
#        plt.figure(20)
#        for i in range(len(imglist)):
#            plt.subplot(1,len(imglist),i+1)
#            plt.title(imglist[i])
#            plt.imshow(cv2.imread(op.join(img_dir,imglist[i])))
#        plt.show()


