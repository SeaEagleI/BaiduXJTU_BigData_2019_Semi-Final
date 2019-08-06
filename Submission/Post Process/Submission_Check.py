# -*- coding: utf-8 -*-
import os,os.path as op
import random
import re

txt1 = '82.0370%.txt'
txt2 = '82.1800%.txt'
#txt1 = '../81.6200%.txt'
#txt2 = '../81.2440%.txt'
turns = 10

CateEstimates = {'001': 30092,
                 '002': 22763,
                 '003': 12753,
                 '004': 1647,
                 '005': 4123,
                 '006': 15671,
                 '007': 5283,
                 '008': 3295,
                 '009': 4370}

def GetLines(txt_path):
    return [i for i in open(txt_path).read().split('\n') if len(i)>0]

def GetCurNum(dir='.'):
    num_list = [eval(re.findall(r'\d+',f)[0]) for f in sorted(os.listdir(dir)) if len(re.findall(r'submission_repl_\d+.txt',f))>0]
    if len(num_list)>0:
        return max(num_list)
    else:
        return 0

def CmpTxt(txt1_path,txt2_path,printable=True):
    cnt = 0
    diffs = []
    txt1_lines = GetLines(txt1_path)
    txt2_lines = GetLines(txt2_path)
    for i in range(len(txt1_lines)):
        if txt1_lines[i]!=txt2_lines[i]:
            cnt += 1    
            id1,ans1 = txt1_lines[i].split('\t')
            id2,ans2 = txt2_lines[i].split('\t')
            if printable:
                print('{}\t{}|{}'.format(id1,ans1,ans2))
            diffs.append([id1,ans1,ans2])
    return diffs

def FHead(fpath):
    filename = fpath.split('/')[-1]
    fhead = '.'.join(filename.split('.')[:-1])
    return fhead.replace('%','')

def MergeName(txt1=txt1,txt2=txt2,tail=''):
    txt1,txt2 = FHead(txt1),FHead(txt2)
    return '{}_{}{}.txt'.format(txt1,txt2,'_'+tail if tail!='' else '')

def LoadDictFromTxt(txt_path):
    ResDict = {}
    lines = [line for line in open(txt_path).read().split('\n') if len(line)>0]
    for line in lines:
        id,ans = line.split('\t')
        ResDict[id] = ans
    return ResDict

def WriteDictToTxt(txt_path,Dict):
    print('[{}]\n'.format(txt_path))
    f = open(txt_path,'w+')
    for id,ans in Dict.items():
        f.writelines('{}\t{}\n'.format(id,ans))
    f.close()
    if op.isfile(txt_path):
        print('Finished: {} Lines'.format(len(Dict.keys())))

def Statistics(Ans_Dict,CateEstimates=CateEstimates):
    Cate_ID = {}
    for id,cate in Ans_Dict.items():
        if cate in Cate_ID.keys():
            Cate_ID[cate].append(id)
        else:
            Cate_ID[cate] = [id]
    Cate_ID = dict(sorted([[cate,idlist] for cate,idlist in Cate_ID.items()],key=lambda x:x[0]))
    for cate,idlist in Cate_ID.items():
        print('{}\t{}\t{}\t{}'.format(cate,len(idlist),CateEstimates[cate],len(idlist)-CateEstimates[cate]))
    print()

def MergeDict(Dict1,Dict2,ModCates):
    Merge_Dict = {}
    _identical,_new_choice,_prior = 0,0,0
    for key,val1 in Dict1.items():
        val2 = Dict2[key]
        if val1==val2:
            Merge_Dict[key] = val1
            _identical += 1
        elif '001' in [val1,val2] and '003' not in [val1,val2] and '005' not in [val1,val2]:
            Merge_Dict[key] = val2 if val1=='001' else val1
            if val2!='001':
                _new_choice += 1
        elif val2 in ModCates:
            Merge_Dict[key] = val2
            _new_choice += 1
        else:
            Merge_Dict[key] = val1
            _prior += 1
    print('\nIdentical:\t\t{}\t{}%\nNew Choice:\t\t{}\t{}%\nPrior Choice:\t\t{}\t{}%\n'.\
          format(_identical,_identical/len(Dict1)*100,\
                 _new_choice,_new_choice/len(Dict1)*100,\
                 _prior,_prior/len(Dict1)*100,))
    return Merge_Dict

def SubmissionGenerator(replace_list,num,base_txt=txt1):
    txt = open(base_txt).read()
    for i in replace_list:
        diff = diffs[i]
        ori,repl = '{}\t{}'.format(diff[0],diff[1]),'{}\t{}'.format(diff[0],diff[2])
        txt = txt.replace(ori,repl)
    open('submission_repl_{}.txt'.format(num),'w+').write(txt)
#    open('rand_list_{}.txt'.format(num),'w+').write(str(replace_list))


Dict1 = LoadDictFromTxt(txt1)
Dict2 = LoadDictFromTxt(txt2)
#Dict3 = LoadDictFromTxt(txt3)
#
#Statistics(Dict1)
#Statistics(Dict2)
#Statistics(Dict3)
#
#priorlist = ['001',
##             '006',
##             '003',
##             '008'
#             ]
#submit_txt = MergeName(txt1,txt2,'{}_MOD'.format('_'.join(priorlist)))
#Merge_Dict = MergeDict(Dict1,Dict2,[])
###Merge_Dict = MergeDict(Dict1,Dict2,['006'])
###Merge_Dict = MergeDict(Dict1,Dict2,['003','008'])
#WriteDictToTxt(submit_txt,Merge_Dict)
#Statistics(Merge_Dict)


diffs = CmpTxt(txt1,txt2,False)
print('Diffs: {}'.format(len(diffs)))
MAX_LEN = len(diffs)
REPL_SIZE = int(MAX_LEN/2)
for i in range(turns):
    REPL_SIZE = random.randint(1,int(MAX_LEN/2))
    repl_list = [random.randint(0,MAX_LEN-1) for i in range(REPL_SIZE)]
    curlist = os.listdir()
    next_num = GetCurNum()+1
    SubmissionGenerator(repl_list,next_num,base_txt=txt1)

