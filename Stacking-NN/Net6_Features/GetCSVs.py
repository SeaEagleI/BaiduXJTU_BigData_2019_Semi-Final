# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

#df = pd.DataFrame({'name': ['Raphael', 'Donatello'],
#                    'mask': ['red', 'purple'],
#                    'weapon': ['sai', 'bo staff']})

Train_Rows = np.load('Train_Rows.npy')[:1000]
Keys = list(np.load('Feature_Keys.npy'))
Columns = list(Train_Rows)
Table = zip(Keys,Columns) 
df = pd.DataFrame(columns=Keys,data=Columns)
df.to_csv('1.csv',index=True)




