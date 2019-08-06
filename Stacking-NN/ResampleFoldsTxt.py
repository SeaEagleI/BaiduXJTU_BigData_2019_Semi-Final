import os,os.path as op
from random import shuffle,sample
import shutil,math

RAW_TXT_DIR       = 'folds_split'
RESAMPLED_TXT_DIR = 'resampled_folds_split'
CLASSES           = ['00{}'.format(i) for i in range(1,10)]

def WriteLinesToTxt(txt_path,lines):
    f = open(txt_path,'w+')
    for line in lines:
        f.writelines(line+'\n')
    f.close()
    if op.isfile(txt_path):
        print('Written {} Lines to {}.'.format(len(lines),txt_path))

def Resample(src_txt,src_dir,dest_dir):
    class_to_filename = {i:[] for i in CLASSES}
    lines = [line.replace('\n','') for line in open(op.join(src_dir,src_txt)).readlines()]
    for line in lines:
        class_to_filename[line[:3]].append(line)
    class_num = [len(class_to_filename[i]) for i in CLASSES]
    max_number = float(max(class_num))
    class_ratio = [max_number/i for i in class_num]

    final_list = []
    for i in range(0,9):
        frac = int((class_ratio[i] - math.floor(class_ratio[i]))*class_num[i])
        integer = int(class_ratio[i])
        final_list += class_to_filename[CLASSES[i]]*integer
        if frac != 0 :
           final_list += sample(class_to_filename[CLASSES[i]], frac)
#        print(frac, integer)
#    print(len(final_list))
    shuffle(final_list)
    WriteLinesToTxt(op.join(dest_dir,'MM_{}'.format(src_txt)),final_list)

def GetResampledFolds(src_dir=RAW_TXT_DIR,dest_dir=RESAMPLED_TXT_DIR):
    train_txt_list = sorted([txt for txt in os.listdir(src_dir) if 'train' in txt])
    val_txt_list   = sorted([txt for txt in os.listdir(src_dir) if 'val' in txt])
    for train_txt in train_txt_list:
        Resample(train_txt,src_dir,dest_dir)
    for val_txt in val_txt_list:
        shutil.copy(op.join(src_dir,val_txt),op.join(dest_dir,val_txt))
    new_val_txt_list   = sorted([txt for txt in os.listdir(dest_dir) if 'val' in txt])
    if new_val_txt_list==val_txt_list:
        print('Moved {} val_txts to {}.'.format(len(new_val_txt_list),dest_dir))

#Get Resampled Folds_Split Train+Val Txts (5 folds) for Net2_1
GetResampledFolds()

