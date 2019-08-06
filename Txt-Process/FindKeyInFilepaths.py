# -*- coding: utf-8 -*-
import os,os.path as op

dir_path = '.'

def GetFilePaths(dir_path='.',paths=[]):
    files = [op.join(dir_path,i) for i in sorted(os.listdir(dir_path)) if op.isfile(op.join(dir_path,i))]
    dirs  = [i for i in sorted(os.listdir(dir_path)) if op.isdir(op.join(dir_path,i))]
    paths += files
    for _dir in dirs:
        paths = GetFilePaths(op.join(dir_path,_dir),paths)
    return paths

def FindKey(lines,key,sep='/'):
    for line in lines:
        if key in line.split(sep)[-1]:
            print(line)

files = GetFilePaths()
FindKey(files,'User_Txts')










