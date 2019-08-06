# -*- coding: utf-8 -*-
from datetime import date,datetime,timedelta
import json

json_path = 'Months Division Criteria.json'
Months = {'2018-Oct':(2018,10),
          '2018-Nov':(2018,11),
          '2018-Dec':(2018,12),
          '2019-Jan':(2019,1),
          '2019-Feb':(2019,2),
          '2019-Mar':(2019,3)}

def GetMonthDates(year,month):   
    dates = []
    cur_dt = date(year,month,1)
    while cur_dt.month==month:
        dates.append(datetime.strftime(cur_dt,'%Y%m%d'))
        cur_dt += timedelta(days=1)
    return sorted(dates)

def WriteJson(dates,label,json_path=json_path):
    dic = json.load(open(json_path))
    for _date in dates:
        dic[_date] = label
    dic = dict(sorted(dic.items(),key=lambda x:x[0]))
    json.dump(dic,open(json_path,'w+'))

json.dump({},open(json_path,'w+'))
for month,span in Months.items():
    Dates = GetMonthDates(span[0],span[1])
    WriteJson(Dates,month)
