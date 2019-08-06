import os,os.path as op
from random import shuffle

TRAIN_IMAGE_DIR = '../train_image_raw'
#TXT_PATH = './train.txt'
CLASSES = ['00{}'.format(i) for i in range(1,10)]
ratio = 0.8

#class_num = [9542, 7538, 3590, 1358, 3464, 5507, 3517, 2617, 2867]
#max_num = float(max(class_num))
#class_ration = [max_num/i for i in class_num]

def WriteTxt(txt_path,lines):
    f = open(txt_path,'w')
    for line in lines:
        f.writelines(line+'\n')

def GetAllImgs(src_dir,txt_path):
    lines = []
    for CLASS in CLASSES:
        lines += ['{}/{}'.format(CLASS,img) for img in sorted(os.listdir(op.join(src_dir,CLASS)))]
    WriteTxt(txt_path,lines)
    if op.isfile(txt_path):
        print('Over')

def GetTrainVal(src_txt,train_txt,val_txt):
    lines = [line[:-1] for line in open(src_txt).readlines()]
    shuffle(lines)
    sep = int(len(lines)*ratio)
    WriteTxt(train_txt,lines[:sep])
    WriteTxt(val_txt,lines[sep:])
    if op.isfile(train_txt) and op.isfile(val_txt):
        print('Over')

GetAllImgs(TRAIN_IMAGE_DIR,'./all.txt')
GetTrainVal('./all.txt','./train.txt','./val.txt')
