# -*- coding: utf-8 -*-
import os,os.path as op
import shutil
import numpy as np
from tqdm import tqdm
from Reshape import Get_Matrix

TRAIN_IMAGE_DIR     = './train_image'
TRAIN_VISIT_TXT_DIR = './train_visit'
TRAIN_VISIT_NPY_DIR = './npy/train_visit'
TEST_IMAGE_DIR      = './test_image'
TEST_VISIT_TXT_DIR  = './test_visit'
TEST_VISIT_NPY_DIR  = './npy/test_visit'

DateList_npy = './DateList.npy'

def RenameDir(src_dir,dest_dir):
    if op.isdir(src_dir) and not op.isdir(dest_dir):
        os.rename(src_dir,dest_dir)
        if not op.isdir(src_dir) and op.isdir(dest_dir):
            print('Renamed Dir: [{} --> {}].'.format(src_dir,dest_dir))

def MakeDir(dest_dir):
    if not op.isdir(dest_dir):
        os.makedirs(dest_dir)
        if op.isdir(dest_dir):
            print('Created Dir: [{}].'.format(dest_dir))

def GetContent(dir_path):
    flist = sorted(os.listdir(dir_path))
    print('{}\t\t{}'.format(dir_path,len(flist)))

def MoveUpWard(dest_dir,dirlist):
    for dir in dirlist:
        src_dir = op.join(dest_dir,dir)
        txtlist = sorted(os.listdir(src_dir))
        print('Moving Up Files: [{} ==> {}]'.format(src_dir,dest_dir))
        for txt in tqdm(txtlist):
            shutil.move(op.join(src_dir,txt),op.join(dest_dir,txt))
        if len(os.listdir(src_dir))==0:
            os.rmdir(src_dir)
            print('Over')

def VisitToArr(src_dir,dest_dir,start=0,unit=15000):
    DateList = list(np.load(DateList_npy))
#    txtlist  = [f[:-4] for f in sorted(os.listdir(src_dir)) if f[-4:]=='.txt']
#    npylist  = [f[:-4] for f in sorted(os.listdir(dest_dir)) if f[-4:]=='.npy']
#    restlist = [f for f in txtlist if f not in npylist]
    restlist  = [f[:-4] for f in sorted(os.listdir(src_dir)) if f[-4:]=='.txt' and not op.isfile(op.join(dest_dir,f[:-4]+'.npy'))]
    print('VisitArrConvertor: [{} ==> {}]'.format(src_dir,dest_dir))
    print('Rest Size: {}'.format(len(restlist)))
    print('[{}]: {} --- {}'.format(start,unit*start,unit*(start+1)))
    for prefix in tqdm(restlist[unit*start:unit*(start+1)]):
        Arr = Get_Matrix(op.join(src_dir,prefix+'.txt'),DateList)
        np.save(op.join(dest_dir,prefix+'.npy'),Arr)
    if len(os.listdir(src_dir))==len(os.listdir(dest_dir)):
        print('Over')

'''
[Step 0]:
Extract all *.tar.gz files to current directory.(Cost: 5.5h)
'''
#Use WINRAR (Win10) to execute this Step.

'''
[Step 1]:
Check File Numbers.(Cost:-)
'''
GetContent('.')
GetContent('./test')
GetContent('./test_part')
GetContent('./train')
GetContent('./train_part')
dir_path = './test_part'
unfinished = [1,5]
for i in range(0,2):
    print('{}\t\t{}'.format(i,len(os.listdir(op.join(dir_path,str(i))))))

'''
[Step 2]:
Move Files to Upper Dir.(Cost: 1.5h==7.5min*12)
'''
MoveUpWard('./test_part', [str(i) for i in range(0,2)])
MoveUpWard('./train_part',[str(i) for i in range(0,10)])

'''
[Step 3]:
Rename Folders & Convert VISIT TXTs to 7x26x24 NPYs.
Muti-Process Running are Recommended.(Cost: about 11h to convert 50w data.)
'''
RenameDir('train',TRAIN_IMAGE_DIR)
RenameDir('train_part',TRAIN_VISIT_TXT_DIR)
RenameDir('test',TEST_IMAGE_DIR)
RenameDir('test_part',TEST_VISIT_TXT_DIR)
MakeDir(TRAIN_VISIT_NPY_DIR)
MakeDir(TEST_VISIT_NPY_DIR)
VisitToArr(TRAIN_VISIT_TXT_DIR,TRAIN_VISIT_NPY_DIR)
VisitToArr(TEST_VISIT_TXT_DIR,TEST_VISIT_NPY_DIR)

'''
[Extra Step]
Eliminate Error NPYs.
'''
#Use ./data/CheckNpy.py to Find Error Npys.
#Found the following Two:
#./data/npy/train_visit/358163_002.npy
#./data/npy/test_visit/010260.npy

#DateList = list(np.load(DateList_npy))
#Arr = Get_Matrix('../Raw Final Data/train_visit/358163_002.txt',DateList)
#np.save('./data/npy/train_visit/358163_002.npy',Arr)
#Arr = Get_Matrix('../Raw Final Data/test_visit/010260.txt',DateList)
#np.save('./data/npy/test_visit/010260.npy',Arr)

