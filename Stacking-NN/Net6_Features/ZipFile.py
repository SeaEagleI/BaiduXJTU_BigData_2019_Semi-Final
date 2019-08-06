#coding:utf-8
import os,os.path as op
import re
import zipfile
from tqdm import tqdm

TRAIN_VISIT_TXT_DIR = '../../Raw Final Data/train_visit'
TEST_VISIT_TXT_DIR  = '../../Raw Final Data/test_visit'

train_txt = '../data/train/train.txt'
val_txt   = '../data/train/val.txt'
npy_path  = 'Rows.npy'

def GetLines(txt_path):
    return [re.findall(r'/(.*?).jpg',line)[0]+'.txt' for line in open(txt_path).read().split('\n') if len(line)>0]

TrainList = GetLines(train_txt)
ValList   = GetLines(val_txt)
#traintxtlist   = sorted(os.listdir(TRAIN_VISIT_TXT_DIR))

def ZipFiles(src_dir,flist,index):
    zip_name = './{}_{}.zip'.format(src_dir,index)
    z = zipfile.ZipFile(zip_name,'w',zipfile.ZIP_DEFLATED)
    for file in tqdm(flist):
        z.write(op.join(src_dir,file), op.join(src_dir.split('/')[-1],file))
    z.close()

def SplitZip(src_dir,flist,cut_num):
    unit = int(len(flist)/cut_num)
    for i in range(cut_num):
        start,end = unit*i,unit*(i+1)
        print('[{}]\n{}-----{}'.format(src_dir,start,end))
        ZipFiles(src_dir,flist[start:end],i+1)

SplitZip(TRAIN_VISIT_TXT_DIR,TrainList[40000:],7)
#SplitZip(TRAIN_VISIT_TXT_DIR,ValList,2)

