# -*- coding: utf-8 -*-
from tqdm import tqdm
import os,os.path as op
import numpy as np
import re,datetime

start_date,end_date = '20181001','20190331'

def Pro_Hours(effect):
    binary = [0 for i in range(24)]
    for hour in effect:
        binary[hour] = 1
    return binary

def Add_List(la,lb):
#    return [la[i]+lb[i] for i in range(len(la))]
    return list(sum(np.array([la,lb])))

def GetBetweenDates(begin_date,end_date):
    date_list = []
    begin_date = datetime.datetime.strptime(begin_date,"%Y%m%d")
    end_date = datetime.datetime.strptime(end_date,"%Y%m%d")
    while begin_date <= end_date:
        date_str = begin_date.strftime("%Y%m%d")
        date_list.append(date_str)
        begin_date += datetime.timedelta(days=1)
    return date_list

def Reshape_Dict(records,DateList):
    result = {}
    for date in DateList:
        result[date] = [0 for i in range(24)]
    for record in records.values():
        for key,val in record.items():
            if key in result.keys():
                result[key] = Add_List(result[key],val)
            else:
                result[key] = val
    return dict([(key,result[key]) for key in sorted(result.keys())])

def Get_Matrix(txt_path,DateList):
    records = {}
    txt = open(txt_path).read()
    lines = [i for i in txt.split('\n') if len(i)>0]
    for line in lines:
        user = line.split('\t')[0]
        days = line.split('\t')[1].split(',')
        record = {}
        for day in days:
            date = day.split('&')[0]
            hours = Pro_Hours([eval(re.sub(r'^0','',hour)) for hour in day.split('&')[1].split('|')])
            record[date] = hours
        records[user] = record
    matrix = list(Reshape_Dict(records,DateList).values())
    return np.array(matrix).reshape(-1,7,24).transpose(1,0,2)

def Print_Arr(array):
    arrlist = list(array)
    for i in range(len(arrlist)):
        arr = arrlist[i]
        if int(np.array(arr.shape).shape[0])>=2:
            Print_Arr(arr)
        else:
            print('[{}]:\n{}'.format(i+1,arr))
    print()


#DateList_npy = 'DateList.npy'
#DateList = GetBetweenDates(start_date,end_date)
#np.save(DateList_npy,np.array(DateList))
#DateList = list(np.load(DateList_npy))

#txt_path = './000011.txt'
#Z = Get_Matrix(txt_path)

#array = np.load('000011.npy')
#Print_Arr(array)
#Print_Arr(Z)
#print(Z==array)
