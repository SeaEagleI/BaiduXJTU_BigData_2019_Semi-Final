# BaiduXJTU_BigData_2019_Semi-Final
# **[Urban Region Function Classification](https://dianshi.baidu.com/competition/30/rank) Top18 Solution**

## Team Brief Intro.
#### Team Name: 
- 浑南摸鱼队

#### Key Members:
- 王德君，东北大学计算机本科大二在读; (Team Leader)
- 姚来刚，东北大学计算机系硕士，研究方向为机器学习与数据挖掘. (Key Teammate)

#### Contest Rankings: 
- Preliminary: Rank 17
- Semi-Final: Rank 18

## Mission Descriptions
给定用户访问数据({Area_ID}.txt)和卫星图片({Area_ID}.jpg)，判断城市用地功能，包括以下9个类别, 并对其分类:
- Residential area
- School
- Industrial park
- Railway station
- Airport
- Park
- Shopping area
- Administrative district
- Hospital  
具体任务描述请见[官网](https://dianshi.baidu.com/competition/30/question) 

## Environmental Requirements

#### OS & GPU Configurations:
- Ubuntu 18.04.1 LTS
- 1080Tix1 + 1060Mx1

#### Python Package Requirements
- Anaconda 4.0.2
- Python 3.6
- Pytorch 1.1.0
- keras 2.2.4
- opencv3
- sklearn
- numpy
- matplotlib

## Detailed Solution
#### Model I: Several Netural Network Stacking (DeepLearning)  
```
#Model Descriptions:
8 Nets
5 folds Stacking
Trained Networks = 7*5+1 (We only trained fold1 on Net6 due to Its Low Local Acc)
```
  
NetWork Name | Baseline | Descriptions | Online Top1-Acc on Test of 5 folds Merged Result
:-:          | :-:      | :-:          | :-:
Net1_raw     | DPN26+Resnext50 | Train with RAW NPYs (40w 182x24 Npys) | 76.21% |
Net2_1       | Net1 | Introduced Resampled Folds_Split for Train | About 76% |
Net3_w       | Net1 | Use Class_Ratio are considered into Loss Calculation | About 76% |
Net4_TTA     | Net1 | Introduced TTA (Test Time Augumentation) | About 76% |
Net5_HR      | Net1 | Introduced HighResample (linear) |  About 76% |
Net6_Features| DenseNet | Introduced Feture Engineering (Features: 175)| About 76% |
Net7_MS      | Net1 | Introduced MultiScale | About 76% |
Net8_MS_cat  | Net1 | Introduced MultiScale & Concatenate | About 76% |
  

#### Model II: Txt Processing
```
1) Txt Identical Check
2) Multivoters based on Total Times A user Appeared in Same Category
3) Multivoters based on Total Hours A user Appeared in Same Category
```
Steps | Content Descriptions | Oringinal Score | After Improved | Precentage Improvement
:-:   | :-:     | :-:             | :-:            | :-: 
1)    | Utilize All identical txts in 






#### Model III: Merge & Rebalance the Predicts in Submissions 

```
Directly Modify Submission.txt:
We Compared 81.2840%.txt and 81.5320%.txt, and found that 001 was TOO MANY (4k more than True Value).
```
- *Category Distributions in Our Submissions (Take 81.6200%.txt for Example)*  

Category | Total Predicted in 81.6200%.txt | Estimated True Value | Difference (Pred-Estimated) 
:-:      | :-:                             | :-:                  | :-:
001 | 34542 | 30092 | +4450 | 
002 | 22026 | 22763 | -737  | 
003 | 13247 | 12753 | +494  |
004 | 1510  | 1647  | -137  |
005 | 4314  | 4123  | +191  |
006 | 12978 | 15671 | -2693 |
007 | 4986  | 5283  | -297  |
008 | 2247  | 3295  | -1048 |
009 | 4150  | 4370  | -220  |


##### Adjustment Measures
```
Therefore, We inplemented a Merge Between the Top2 Submissions.
Based on 81.6200%.txt, whenever it predicts to 001 in 81.5320%.txt and & 00x (x!=3 and x!=5, as 003 and 005 are More-Predicted), we choose 00x as the answer.
```





