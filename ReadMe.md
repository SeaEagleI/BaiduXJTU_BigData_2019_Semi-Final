# BaiduXJTU_BigData_2019_Semi-Final
# **Urban Region Function Classification [Top18](https://dianshi.baidu.com/competition/30/rank) Solution**

---
## Competetition Intro.
- [IKCEST首届“一带一路”国际大数据竞赛](http://www.ikcest.org/bigdata2019/)
- [赛题：基于遥感影像和用户行为的城市区域功能分类](https://dianshi.bce.baidu.com/competition/30/rule)
- [回忆录 - IKCEST首届“一带一路”国际大数据竞赛(2019)获奖经历](https://blog.csdn.net/Flying_Dutch/article/details/106134801)

## Team Brief Intro.
#### Team Name: 
- 浑南摸鱼队

#### Key Members:
- 王德君，东北大学计算机本科大二在读; (Team Leader)
- 姚来刚，东北大学计算机系硕士，研究方向为机器学习与数据挖掘. (Key Teammate)

#### Contest Rankings: 
- Preliminary: Rank 17
- Semi-Final: Rank 18

---
## Mission Descriptions

Build models to classify the functions of urban areas with data of satellite images({Area_ID}.jpg) and user behavior({Area_ID}.txt) from given geographical areas.  

- Tables of the functions of urban areas:

CategoryID | Functions of Areas
:-:        | :-:
001	| Residential area
002	| School
003	| Industrial park
004	| Railway station
005	| Airport
006	| Park
007	| Shopping area
008	| Administrative district
009	| Hospital

For more Detailed Task descriptions, please go to [赛题详情](https://dianshi.baidu.com/competition/30/question)  

---
## Environmental Requirements

#### OS & GPU Configurations
- Ubuntu 18.04.1 LTS
- GTX 1080Ti x 1 + GTX 1060M x 1
- Baidu AI Studio (Tesla V100 x 36, For Model I 36 Networks Training)

#### Python Package Requirements
- Anaconda 4.7.10
- python 3.6
- pytorch 1.1.0
- keras 2.2.4
- opencv3
- sklearn
- numpy
- matplotlib

---
## Submission Timeline
Model     | Baseline Acc | Top Result
:-:       | :-:          | :-:       
Model I   | None         | 77.08%
Model II  | 77.08%       | [81.6200%](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Submission/81.6200%25.txt)
Model III | [81.6200%](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Submission/81.6200%25.txt)<br>[81.2440%](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Submission/Post%20Process/81.2440%25.txt)       | [82.1800%](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Submission/Post%20Process/82.1800%25.txt)

## Detailed Solution
### Model I: Several Netural Network Stacking ([DeepLearning](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/tree/master/Stacking-NN))  
```
# Model Descriptions:
8 Nets
5 folds Stacking
Trained Networks = 7*5+1 (We only trained fold1 on Net6 because of Its Low Local Acc).
# Result:
After 36 NN Stacking, we reached a top online acc of 77.08%.
```  

NetWork Name | Baseline | Descriptions | Online Top1-Acc on Test of 5 folds Merged Result
:-:          | :-:      | :-:          | :-:
[Net1_raw](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Stacking-NN/folds_py/Net1_raw_fold1.py)     | DPN26+Resnext50 | Train with RAW NPYs (40w 182x24 Npys) | 76.21% |
[Net2_1](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Stacking-NN/folds_py/Net2_1_fold1.py)      | Net1 | Introduced Resampled Folds_Split for Train | About 76% |
[Net3_w](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Stacking-NN/folds_py/Net3_w_fold1.py)       | Net1 | Class_Ratio were considered into Loss Calculation | About 76% |
[Net4_TTA](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Stacking-NN/folds_py/Net4_TTA_fold1.py)     | Net1 | Introduced TTA (Test Time Augumentation) | About 76% |
[Net5_HR](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Stacking-NN/folds_py/Net5_HR_fold1.py)      | Net1 | Introduced HighResample (linear) |  About 76% |
[Net6_Features](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Stacking-NN/Net6_Features)| DenseNet | Introduced Feture Engineering (Features: 175)| 61.19% |
[Net7_MS](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Stacking-NN/folds_py/Net7_MS_fold1.py)      | Net1 | Introduced MultiScale | About 76% |
[Net8_MS_cat](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Stacking-NN/folds_py/Net8_MS_cat_fold1.py)  | Net1 | Introduced MultiScale & Concatenate | About 76% |

### Model II: Txt Processing ([Feature Engineering](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Txt-Process))
```
# Steps:
1) Txt Identical Check (Completed)
2) Multivoters based on Total Times A user Appeared in Same Category (Completed)
3) Multivoters based on Total Hours A user Appeared in Same Category (Only 3/2000 json files were processed)
# Result:
After above 3 steps, we got a submission of 81.62%. (81.6200%.txt)
# Notes:
[1] Step 3 was not completed Because of Limited Time and Computation Resources, only 3/2000 data was processed.
[2] In this project we abbreviate {Preliminary,Semi-Final}-{Train,Test}-Datasets as {P,S}{Tr,Te} ==> {PTr,PTe,STr,STe}.
```
Steps | Content Descriptions | Oringinal Score | After Improved | Source Code
:-:   | :-:                  | :-:             | :-:            | :-:
(1)   | Utilize Identical txts' Categories in PTr & STr to provide answers for STe | 77.04% | 78.74% | [SelfDuplicateCheck](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Txt-Process/SelfDuplicateCheck.py)
(2)   | Multivoters based on Total Times A user Appeared in Same Category | 78.74% | 81.62% | [MergeVotes](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Txt-Process/MergeVotes.py)
(3)   | Multivoters based on Total Hours A user Appeared in Same Category | 81.62% | - | [AdvM2_train](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Txt-Process/AdvM2_train.py)

### Model III: Merge & Rebalance the Predicts in Submissions ([Post-processing](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Submission/Post%20Process)）
```
Directly Modify Submission.txt:
We Compared predicts in 81.2440%.txt and 81.6200%.txt, finding that 001 was TOO MANY (4k more than True Value), 003/005 were a bit more-predicted, and others were all less-predicted.
```
- *Category Distributions in Our Submissions (Take [81.6200%.txt](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Submission/81.6200%25.txt) for Example)*  

Category | Total Predicts in [81.6200%.txt](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Submission/81.6200%25.txt) | Estimated True Value | Difference (Pred-Estimated) | Evaluation
:-:      | :-:                            | :-:                  | :-:                         | :-:
001 | 34542 | 30092 | +4450 | Too Much More
002 | 22026 | 22763 | -737  | Much Less
003 | 13247 | 12753 | +494  | More
004 | 1510  | 1647  | -137  | Little Less
005 | 4314  | 4123  | +191  | More
006 | 12978 | 15671 | -2693 | Much Less
007 | 4986  | 5283  | -297  | Little Less
008 | 2247  | 3295  | -1048 | Much Less
009 | 4150  | 4370  | -220  | Little Less

```
Therefore, We Merged the Predicts among our ex-Top2 Submissions. (81.6200%.txt & 81.2440%.txt)
Strategy & Rules:
1) Compare & Merge the predicts in the two txt file by Replacing those '001's to other less-predicted categories.
2) While the two gives the same prediction or both predictions are in More-Predicted Categories ['001','003','005'], Choose the answer in 81.6200%.txt as result Beacause of its Higher Acc.
After this operation, we got our final best submission 82.1800%.txt, which reached 82.18%.
```
- Related Source Code ([Submission_Check](https://github.com/SeaEagleI/BaiduXJTU_BigData_2019_Semi-Final/blob/master/Submission/Post%20Process/Submission_Check.py))
```python3
def MergeDict(Dict1,Dict2,ModCates):
    Merge_Dict = {}
    _identical,_new_choice,_prior = 0,0,0
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
    return Merge_Dict

txt1 = '../81.6200%.txt'
txt2 = '../81.2440%.txt'
Dict1 = LoadDictFromTxt(txt1)
Dict2 = LoadDictFromTxt(txt2)

priorlist = ['001',
#             '006',
#             '003',
#             '008'
            ]
submit_txt = MergeName(txt1,txt2,'{}_MOD'.format('_'.join(priorlist)))
Merge_Dict = MergeDict(Dict1,Dict2,[])
##Merge_Dict = MergeDict(Dict1,Dict2,['006'])
##Merge_Dict = MergeDict(Dict1,Dict2,['003','008'])
WriteDictToTxt(submit_txt,Merge_Dict)
Statistics(Merge_Dict)
```

## 中文回忆录

### 摘要
作为队长，我在2019年5月20日-2019年8月1日，共计两个半月的时间里率团队（几个本校的研究生）参加了百度大数据竞赛，并获得了18/2312（top0.78%）的好成绩。此次比赛的主要内容是城市区域功能分类，是个多模态分类任务，内容包括图像识别和文本特征挖掘。
在参与这次比赛的过程中，我们队经历了不少挫折，也获得了不少收获。我们最终提出了图像-文本融合网络识别模型和基于文本特征的投票器，拿到了初赛第17，复赛第18的成绩。

### 实现经历
我们首先从图片入手，用ResNeXt作为模型喂入图片，最后发现准确率保持在50%左右，效果不是很理想。
随后，我们仔细地检查了一遍图片，发现大约20%的遥感图片中雾霾严重，甚至存在大面积黑块，于是我们进行了一遍图片清洗，把含有黑块的图片从数据集中去除，同时对剩余的图片进行dehaze，这样一来，纯图片的准确率达到了55%。
之后，我们把文本的时间序列信息提取出来，转化成128×24的图片，用DPN26进行训练。把图片和文本的识别器输出结果concat后喂入fc层，这也就是Net1的结构，此时的综合识别准确率为64%。
接着，我们开始用stacking和集成学习的方法来训练数据。通过对图片进行 TTA、缩放、上下采样、加权，以及对文本进行特征提取等各种操作，我们获得了其他六个网络模型：Net2~Net7。同时我们把训练集数据分成5份进行交叉训练和识别，并把第一步网络获得的结果加权平均后用Xgboost进行二次训练。这一步之后识别的准确率提高到了76%。
此时，我们发现自己和排名靠前的其他队伍在准确率上还有一定差距，同时结合bbs的信息。我们发现：用户对于不同地区的访问记录间存在一些关联，而这种关联是通过用户ID产生的，而此类信息通过时间序列模型是挖掘不到的。
于是我们创造性地提出了基于单个用户在某地区出现次数的投票器，用于进一步挖掘文本特征，最终将准确率提高到了81.62%。虽然此后我们又提出了一个基于小时数的更强的投票器，但由于算力和时间的限制，没能实现。
通过人为对结果进行一定调整，我们队最终拿到了82.18%，也就是第18名的成绩。没能获得更好的名次，还是有一些遗憾。

### 结语
这次比赛经历不仅提高了我的动手能力，促进了我的自学能力，培养了我对机器学习相关领域的兴趣。也让我学会了不少理论知识和实践技巧：在此期间，我对神经网络调参的相关技巧、常用的集成学习方法、主流的CNN网络框架、和文本特征工程的一般流程有了较为深刻的了解和掌握。这为我未来从事相关领域的学术研究打下了坚实的和实践基础。
