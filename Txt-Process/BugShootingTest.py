# -*- coding: utf-8 -*-
import os,os.path as op
from tqdm import tqdm
import re,ujson
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from collections import defaultdict

user_txts_path = 'dic.json'
CLASS = '005'
ZEROES = [0 for i in range(9)]

def AppendDict(Dict,key,index,val):
    if key not in Dict.keys():
        Dict[key] = [0 for i in range(9)]
    Dict[key][index] = val

User_Txts = {}
#user_txts = ujson.load(open(user_txts_path))
for user,txtlist in user_txts.items():
    AppendDict(User_Txts,user,int(CLASS)-1,len(txtlist))
#    if user not in User_Txts.keys():
#        User_Txts[user] = [0 for i in range(9)]
#    _list = User_Txts[user]
#    _list[int(CLASS)-1] = len(txtlist)
#    print(_list)
#    User_Txts[user] = txtlist

print(User_Txts)

