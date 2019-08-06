# -*- coding: utf-8 -*-
import numpy as np

txt_path = './data/train/val.txt'
npy_path = './targets.npy'

#table = dict()
lines = [i for i in open(txt_path).read().split('\n') if len(i)>0]
targets = [eval(line.split('/')[0][-1])-1 for line in lines]
#print(targets[:500])

np.save(npy_path,np.array(targets))


