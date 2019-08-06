# -*- coding: utf-8 -*-
from datetime import date,datetime,timedelta
import json

json_path = 'Date Attributes Criteria.json'

def GetWeekends(start_dt,end_dt):   
    weekends = []
    #Generate All Saturdays
    cur_dt = start_dt
    cur_dt += timedelta(days = 5-cur_dt.weekday())  
    while cur_dt<=end_dt:
        weekends.append(datetime.strftime(cur_dt,'%Y%m%d'))
        cur_dt += timedelta(days=7)
    #Generate All Sundays
    cur_dt = start_dt
    cur_dt += timedelta(days = 6-cur_dt.weekday())
    while cur_dt<=end_dt:
        weekends.append(datetime.strftime(cur_dt,'%Y%m%d'))
        cur_dt += timedelta(days=7)
    return sorted(weekends)

def GetAllDates(start_dt,end_dt):   
    dates = []
    cur_dt = start_dt
    while cur_dt<=end_dt:
        dates.append(datetime.strftime(cur_dt,'%Y%m%d'))
        cur_dt += timedelta(days=1)
    return sorted(dates)

def WriteJson(dates,label,json_path=json_path):
    dic = json.load(open(json_path))
    for _date in dates:
        dic[_date] = label
    dic = dict(sorted(dic.items(),key=lambda x:x[0]))
    json.dump(dic,open(json_path,'w+'))

NationalDays = []
for i in range(1, 8):
    if(i<10):
        NationalDays.append('2018100'+str(i))
    else:
        NationalDays.append('201810'+str(i))
NewYearsDays = ['20181230', '20181231', '20190101']
SpringFestival = []
for i in range(4, 11):
    if(i<10):
        SpringFestival.append('2019020'+str(i))
    else:
        SpringFestival.append('201902'+str(i))
ExtraWorkdays = ['20181229','20190202','20190203']


Dates    = GetAllDates(date(2018,10,1),date(2019,3,31))
Weekends = GetWeekends(date(2018,10,1),date(2019,3,31))

json.dump({},open(json_path,'w+'))
WriteJson(Dates,'workday')
WriteJson(Weekends,'weekend')
WriteJson(ExtraWorkdays,'workday')
WriteJson(NationalDays,'national-day')
WriteJson(NewYearsDays,'new-years-day')
WriteJson(SpringFestival,'spring-festival')
