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
- Baidu AI Studio (For Model I 36 Networks Training)

#### Python Package Requirements
- Anaconda 4.0.2
- Python 3.6
- Pytorch 1.1.0
- keras 2.2.4
- opencv3
- sklearn
- numpy
- matplotlib

## Submission Timeline
Model     | Baseline Acc | Top Result
:-:       | :-:          | :-:       
Model I   | None         | 77.08%
Model II  | 77.08%       | [81.6200%](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Submission/81.6200%25.txt)
Model III | 78.74%       | [82.1800%](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Submission/Post%20Process/82.1800%25.txt)

## Detailed Solution
### Model I: Several Netural Network Stacking ([DeepLearning](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/tree/master/Stacking-NN))  
```
#Model Descriptions:
8 Nets
5 folds Stacking
Trained Networks = 7*5+1 (We only trained fold1 on Net6 due to Its Low Local Acc)
```  

NetWork Name | Baseline | Descriptions | Online Top1-Acc on Test of 5 folds Merged Result
:-:          | :-:      | :-:          | :-:
[Net1_raw](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Stacking-NN/folds_py/Net1_raw_fold1.py)     | DPN26+Resnext50 | Train with RAW NPYs (40w 182x24 Npys) | 76.21% |
[Net2_1](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Stacking-NN/flods_py/Net2_1_fold1.py)      | Net1 | Introduced Resampled Folds_Split for Train | About 76% |
[Net3_w](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Stacking-NN/flods_py/Net3_w_fold1.py)       | Net1 | Use Class_Ratio are considered into Loss Calculation | About 76% |
[Net4_TTA](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Stacking-NN/flods_py/Net4_TTA_fold1.py)     | Net1 | Introduced TTA (Test Time Augumentation) | About 76% |
[Net5_HR](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Stacking-NN/flods_py/Net5_HR_fold1.py)      | Net1 | Introduced HighResample (linear) |  About 76% |
[Net6_Features](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Stacking-NN/Net6_Features)| DenseNet | Introduced Feture Engineering (Features: 175)| 61.19% |
[Net7_MS](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Stacking-NN/flods_py/Net7_MS_fold1.py)      | Net1 | Introduced MultiScale | About 76% |
[Net8_MS_cat](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Stacking-NN/flods_py/Net8_MS_cat_fold1.py)  | Net1 | Introduced MultiScale & Concatenate | About 76% |
  
### Model II: Txt Processing ([Feature Engineering](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Txt-Process))
```
1) Txt Identical Check
2) Multivoters based on Total Times A user Appeared in Same Category
3) Multivoters based on Total Hours A user Appeared in Same Category
(Step 3 was not completed Because of Limited Computational Resources and Time, only 3/2000 data was processed)
Notes. Abbreviate {Preliminary,Semi-Final}-{Train,Test}-Datasets as {P,S}{Tr,Te} ==> {PTr,PTe,STr,STe}
```
Steps | Content Descriptions | Oringinal Score | After Improved | Source Code
:-:   | :-:                  | :-:             | :-:            | :-:
(1)   | Utilize Identical txts' Categories in PTr & STr to provide answers for STe | 77.04% | 78.74% | [SelfDuplicateCheck](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Txt-Process/SelfDuplicateCheck.py)
(2)   | Multivoters based on Total Times A user Appeared in Same Category | 78.74% | 81.62% | [MergeVotes](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Txt-Process/MergeVotes.py)
(3)   | Multivoters based on Total Hours A user Appeared in Same Category | 81.62% | - | [AdvM2_train](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Txt-Process/AdvM2_train.py)

### Model III: Merge & Rebalance the Predicts in Submissions ([Post-processing](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Submission/Post%20Process)）
```
Directly Modify Submission.txt:
We Compared 81.2840%.txt and 81.5320%.txt, and found that 001 was TOO MANY (4k more than True Value).
```
- *Category Distributions in Our Submissions (Take [81.6200%.txt](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Submission/81.6200%25.txt) for Example)*  

Category | Total Predicts in [81.6200%.txt](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Submission/81.6200%25.txt) | Estimated True Value | Difference (Pred-Estimated) | Evaluation
:-:      | :-:                            | :-:                  | :-:                         | :-:
001 | 34542 | 30092 | +4450 | Too Much More
002 | 22026 | 22763 | -737  | Much Less
003 | 13247 | 12753 | +494  | More
004 | 1510  | 1647  | -137  | Almost Accurate
005 | 4314  | 4123  | +191  | Almost Accurate
006 | 12978 | 15671 | -2693 | Much Less
007 | 4986  | 5283  | -297  | Almost Accurate
008 | 2247  | 3295  | -1048 | Much Less
009 | 4150  | 4370  | -220  | Almost Accurate

```
Therefore, We Merge the Predicts among our ex-Top Submissions. (81.6200%.txt & 81.2440%.txt)
Based on 81.6200%.txt, whenever it predicts to 001 in 81.5320%.txt and & 00x (x!=3 and x!=5, as 003 and 005 are More-Predicted), we choose 00x as the answer.
```
- Related Source Code ([Submission_Check](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Submission/Post%20Process/Submission_Check.py))
```
Dict1 = LoadDictFromTxt('81.6200%.txt')
Dict2 = LoadDictFromTxt('81.2440%.txt')

for key,val1 in Dict1.items():
    val2 = Dict2[key]
    if val1==val2:
        Merge_Dict[key] = val1
    elif '001' in [val1,val2] and '003' not in [val1,val2] and '005' not in [val1,val2]:
        Merge_Dict[key] = val2 if val1=='001' else val1
    elif val2 in ModCates:
        Merge_Dict[key] = val2
    else:
        Merge_Dict[key] = val1
```
