# -*- coding: utf-8 -*-
import json

periods_path    = 'Periods Division Criteria.json'
dates_path      = 'Date Attributes Criteria.json'
months_path     = 'Months Division Criteria.json'
periods         = json.load(open(periods_path))
#period_strs     = sorted(set(periods.values()))
date_attributes = json.load(open(dates_path))
#date_strs       = sorted(set(date_attributes.values()))
months          = json.load(open(months_path))
#month_strs      = sorted(set(months.values()))
period_strs     = ['early-morning',
                   'morning',
                   'noon',
                   'afternoon',
                   'evening', 
                   'night',
                   'midnight']
date_strs       = ['workday',
                   'weekend',
                   'national-day',
                   'new-years-day',
                   'spring-festival']
month_strs      = ['2018-Oct', 
                   '2018-Nov', 
                   '2018-Dec', 
                   '2019-Jan', 
                   '2019-Feb', 
                   '2019-Mar']

def AppendList(dic,key,val):
    if key not in dic.keys():
        dic[key] = [val]
    else:
        dic[key].append(val)

def SplitHours(hours):
    slist,period = [],[hours[0]]
    for i in range(1,len(hours)):
        if hours[i]-hours[i-1]==1:
            period.append(hours[i])
        else:
            slist.append(period)
            period = [hours[i]]
    slist.append(period)
    result = []
    for period in slist:
        result.append([period[0],period[-1]-period[0],period[-1]])
    return result

def Statistics(dic):
    reversed_dic = {}    
    for key,val in dic.items():
        AppendList(reversed_dic,val,key)
    return reversed_dic

def GetFeatures(txt_path,label):
    lines = [i for i in open(txt_path).read().split('\n') if len(i)>0]
    UDT,DU = {},{}
    Enter,Exit = {},{}
    pstrs = period_strs+date_strs+month_strs
    for pstr in pstrs:
        Enter[pstr],Exit[pstr] = {},{}
    for pstr in pstrs:
        for qstr in pstrs:
            Enter[pstr][qstr],Exit[pstr][qstr] = [],[]
    for line in lines:
        user,days_log = line.split('\t')
        days = days_log.split(',')
        dates = {}
    #    print('[{}] {}'.format(lines.index(line)+1,user))
        for day in days:
            date,hours_log = day.split('&')
            AppendList(DU,date,user)
            hours = [int(hour) for hour in hours_log.split('|')]
            dates[date] = SplitHours(hours)
            for row in dates[date]:
                AppendList(Enter[periods[str(row[0])]],date,user)
                AppendList(Exit[periods[str(row[-1])]],date,user)
                AppendList(Enter[date_attributes[date]],periods[str(row[0])],user)
                AppendList(Exit[date_attributes[date]],periods[str(row[-1])],user)
                AppendList(Enter[months[date]],periods[str(row[0])],user)
                AppendList(Exit[months[date]],periods[str(row[-1])],user)
        UDT[user] = dates

    # create feature dict
    features = {}
    date_stats = Statistics(date_attributes)
    
    #Aera Category ID 
    #Label: 1 variable
    features['label'] = label

    #General Statistics of Total Nubmers.
    #3 variables
    features['Total Number of Days Counted']   = 182
    features['Total Number of Users Appeared'] = len(UDT)
    features['Total Person-Time of Visit']     = len(DU)
    
    #Date-User Accumlation
    max_dates, min_dates, added_dates = 0, 2**40, 0
    for user in UDT.keys():
        l = len(UDT[user].keys())
        added_dates += l
        if l>max_dates:
            max_dates = l
        if l<min_dates:
            min_dates = l
    features['Max of Days the Same User Appeared'] = max_dates
    features['Min of Days the Same User Appeared'] = min_dates
    features['Avg of Days the Same User Appeared'] = added_dates/features['Total Number of Users Appeared']
    
    #User-Date Accumlation
    max_users, min_users, added_users = 0, 2**40, 0
    for date in DU.keys():
        l = len(DU[date])
        added_users += l
        if l > max_users:
            max_users = l
        if l < min_users:
            min_users = l
    features['Max of Different Users Appeared in A Day'] = max_users
    features['Min of Different Users Appeared in A Day'] = min_users
    features['Avg of Different Users Appeared in A Day'] = added_users/features['Total Number of Days Counted']

    #Total Person-Time in Visit Log
    features['Total Users'] = added_users
    #Position reserved for Person-Time Accumulations
    #...

    #Total Number of 
    #Entering & Exiting Person-Time in 
    #Each Specified Time Period of 
    #A Day.
    #7[periods]*2[directions] = 14 variables
    for period in period_strs:
        features[period+' '+"Total-Person-Time-Enter"] = sum([len(Enter[period][date]) for date in Enter[period].keys()])
        features[period+' '+"Total-Person-Time-Exit"]  = sum([len(Exit[period][date]) for date in Exit[period].keys()])

    #Total Number & Average per day of 
    #Entering & Exiting Person-Time in 
    #Each Specified Time Period of 
    #Each Date_Attributes.
    #5[Date_Attributes]*7[periods]*2[directions]*2[calculation forms] = 140 variables
#    exit()
    for ds in date_strs:
        for ps in period_strs:
            features[ds+' '+ps+' '+"Total-Person-Time-Enter"] = len(Enter[ds][ps])
            features[ds+' '+ps+' '+"Avg-Person-Time-Enter"]   = len(Enter[ds][ps]) / len(date_stats[ds])
            features[ds+' '+ps+' '+"Total-Person-Time-Exit"]  = len(Exit[ds][ps])
            features[ds+' '+ps+' '+"Avg-Person-Time-Exit"]    = len(Exit[ds][ps]) / len(date_stats[ds])

    #Total Number of 
    #Entering & Exiting Person-Time in 
    #Each Months.
    #6[Months]*2[directions] = 12 variables
    for month in month_strs:
        features[month+' '+"Total-Person-Time-Enter"] = sum([len(Enter[month][period]) for period in Enter[month].keys()])
        features[month+' '+"Total-Person-Time-Exit"]  = sum([len(Exit[month][period]) for period in Exit[month].keys()])
    return features

features = GetFeatures('./000012_002.txt','002')
#labels = list(features.keys())
