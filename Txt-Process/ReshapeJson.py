# -*- coding: utf-8 -*-
from tqdm import tqdm
import os,os.path as op

PTRAIN_DIR = 'PTrain_User_Txts'
STRAIN_DIR = 'STrain_User_Txts'
STEST_DIR  = 'STest_User_Txts'
MERGE_DIR  = 'Merge_Research'
ANS_DIR    = 'Ans_Dicts'

def Operate(line):
    user,txtlist = line.split(':')
    return '{"_id":'+user+',"txtlist":'+txtlist+'}\n'

def ReshapeDict(src_json):
    dest_json = '{}_Reshaped.json'.format(src_json.split('.')[0])
    raw_json = open(src_json).read()[1:-1]+','
    new_json = ''.join([Operate(line+']]') for line in tqdm(raw_json.split(']],')[:-1])])
    open(dest_json,'w+').write(new_json)
    if op.isfile(dest_json):
        print('Over')


#raw_json_path = 'test_raw.json'
#new_json_path = 'test.json'

PTr_User_Txts_path = op.join(PTRAIN_DIR,'PTr_User_Txts.json')
ReshapeDict(PTr_User_Txts_path)

STr_User_Txts_path = op.join(STRAIN_DIR,'STr_User_Txts.json')
ReshapeDict(STr_User_Txts_path)








