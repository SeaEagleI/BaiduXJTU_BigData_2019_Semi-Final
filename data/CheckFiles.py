# -*- coding: utf-8 -*-
import numpy as np
import os,os.path as op
from tqdm import tqdm

TRAIN_IMAGE_DIR     = './train_image_raw'
TEST_IMAGE_DIR      = './test_image_raw'
TRAIN_VISIT_NPY_DIR = './npy/train_visit'
TEST_VISIT_NPY_DIR  = './npy/test_visit'

train_txt = './train/train.txt'
val_txt = './train/val.txt'

def CheckNpyList(dir_path):
    npylist = sorted(os.listdir(dir_path))
    print('[{} ----- {}]:'.format(dir_path,len(npylist)))
    for npy in tqdm(npylist):
        try:
            np.load('{}/{}'.format(dir_path,npy))
        except:
            print('{}'.format(npy))
    print()

def CheckImgList(txt_path,image_dir):
    imglist = sorted(open(txt_path).read().split('\n'))
    CLASSES = sorted(os.listdir(image_dir))
    flist = []
    for CLASS in CLASSES:
        flist += ['{}/{}'.format(CLASS,f) for f in os.listdir(image_dir+'/'+CLASS)]
    flist = sorted(flist)
    for img in tqdm(imglist):
        if img not in flist:
            print(img)

#Two Inconsistent NPY FILE are Found:
#./npy/train_visit/358163_002.npy
#./npy/test_visit/010260.npy

'''Check Npys'''
CheckNpyList(TRAIN_VISIT_NPY_DIR)
CheckNpyList(TEST_VISIT_NPY_DIR)

'''Check Images'''
CheckImgList(train_txt,TRAIN_IMAGE_DIR)
CheckImgList(val_txt,TRAIN_IMAGE_DIR)

