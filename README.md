# BaiduXJTU_BigData_2019_Semi-Final  

## Model I: DeepLearning  
`(8 Nets, 5 folds Stacking, Training Turns = 7*5+1 (We only trained fold1 on Net6 dued to Its Low Local Acc)`  

NetWork Name | Baseline | Descriptions | Purpose | Local Top1 Acc on Folds_Val | 444
:-:          | :-:      | :-:          | :-:     | :-:                         | :-:
Net1_raw     | DPN26+Resnext50 | Train with RAW NPYs (40w 182x24 Npys)| SOA models ==> SOA results | ddd | eee| |
Net2_1       | Net1 | Introduced Resampled Folds_Split for Train | Cope with Unbalanced Categories.| hhh | iii | 000||
Net3_w       | Net1 | Use Class_Ratio are considered into Loss Calculation | Unbiased Training | iii | 000||
Net4_TTA     | Net1 | Introduced TTA (Test Time Augumentation) | hhh | iii | 000||
Net5_HR      | Net1 | Introduced HighResample (linear) | hhh | iii | 000||
Net6_Features| Net1 | Introduced HighResample (linear) | hhh | iii | 000||
Net7_MS      | Net1 | Introduced HighResample (linear) | hhh | iii | 000||
Net8_MS_cat  | Net1 | Introduced HighResample (linear) | hhh | iii | 000||



