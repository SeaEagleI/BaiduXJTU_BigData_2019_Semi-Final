# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
import torch

CLASSES = ['00{}'.format(i) for i in range(1,10)]

#Single Network Val/Test Npzs
Net1_f1_Val, Net1_f1_Test = np.load("Net1_raw_fold1_Val.npz"), np.load("Net1_raw_fold1_Test.npz")
Net1_f2_Val, Net1_f2_Test = np.load("Net1_raw_fold2_Val.npz"), np.load("Net1_raw_fold2_Test.npz")
Net1_f3_Val, Net1_f3_Test = np.load("Net1_raw_fold3_Val.npz"), np.load("Net1_raw_fold3_Test.npz")
Net1_f4_Val, Net1_f4_Test = np.load("Net1_raw_fold4_Val.npz"), np.load("Net1_raw_fold4_Test.npz")
Net1_f5_Val, Net1_f5_Test = np.load("Net1_raw_fold5_Val.npz"), np.load("Net1_raw_fold5_Test.npz")

Net2_f1_Val, Net2_f1_Test = np.load("Net2_1_fold1_Val.npz"),np.load("Net2_1_fold1_Test.npz")
#Net2_f2_Val, Net2_f2_Test = np.load("Net2_1_fold2_Val.npz"),np.load("Net2_1_fold2_Test.npz")
#Net2_f3_Val, Net2_f3_Test = np.load("Net2_1_fold3_Val.npz"),np.load("Net2_1_fold3_Test.npz")
#Net2_f4_Val, Net2_f4_Test = np.load("Net2_1_fold4_Val.npz"),np.load("Net2_1_fold4_Test.npz")
#Net2_f5_Val, Net2_f5_Test = np.load("Net2_1_fold5_Val.npz"),np.load("Net2_1_fold5_Test.npz")

Net3_f1_Val, Net3_f1_Test = np.load("Net3_w_fold1_Val.npz"),np.load("Net3_w_fold1_Test.npz")
#Net3_f2_Val, Net3_f2_Test = np.load("Net3_w_fold2_Val.npz"),np.load("Net3_w_fold2_Test.npz")
#Net3_f3_Val, Net3_f3_Test = np.load("Net3_w_fold3_Val.npz"),np.load("Net3_w_fold3_Test.npz")
#Net3_f4_Val, Net3_f4_Test = np.load("Net3_w_fold4_Val.npz"),np.load("Net3_w_fold4_Test.npz")
#Net3_f5_Val, Net3_f5_Test = np.load("Net3_w_fold5_Val.npz"),np.load("Net3_w_fold5_Test.npz")

Net4_f1_Val, Net4_f1_Test = np.load("Net4_TTA_fold1_Val.npz"),np.load("Net4_TTA_fold1_Test.npz")
#Net4_f2_Val, Net4_f2_Test = np.load("Net4_TTA_fold2_Val.npz"),np.load("Net4_TTA_fold2_Test.npz")
#Net4_f3_Val, Net4_f3_Test = np.load("Net4_TTA_fold3_Val.npz"),np.load("Net4_TTA_fold3_Test.npz")
#Net4_f4_Val, Net4_f4_Test = np.load("Net4_TTA_fold4_Val.npz"),np.load("Net4_TTA_fold4_Test.npz")
#Net4_f5_Val, Net4_f5_Test = np.load("Net4_TTA_fold5_Val.npz"),np.load("Net4_TTA_fold5_Test.npz")

Net5_f1_Val, Net5_f1_Test = np.load("Net5_HR_fold1_Val.npz"), np.load("Net5_HR_fold1_Test.npz")
Net5_f2_Val, Net5_f2_Test = np.load("Net5_HR_fold2_Val.npz"), np.load("Net5_HR_fold2_Test.npz")
Net5_f3_Val, Net5_f3_Test = np.load("Net5_HR_fold3_Val.npz"), np.load("Net5_HR_fold3_Test.npz")
Net5_f4_Val, Net5_f4_Test = np.load("Net5_HR_fold4_Val.npz"), np.load("Net5_HR_fold4_Test.npz")
Net5_f5_Val, Net5_f5_Test = np.load("Net5_HR_fold5_Val.npz"), np.load("Net5_HR_fold5_Test.npz")

Net6_f1_Val, Net1_f1_Test = np.load("Net6_Features_fold1_Val.npz"), np.load("Net6_Features_fold1_Test.npz")

Net7_f1_Val, Net7_f1_Test = np.load("Net7_MS_fold1_Val.npz"),np.load("Net7_MS_fold1_Test.npz")
#Net7_f2_Val, Net7_f2_Test = np.load("Net7_MS_fold2_Val.npz"),np.load("Net7_MS_fold2_Test.npz")
#Net7_f3_Val, Net7_f3_Test = np.load("Net7_MS_fold3_Val.npz"),np.load("Net7_MS_fold3_Test.npz")
#Net7_f4_Val, Net7_f4_Test = np.load("Net7_MS_fold4_Val.npz"),np.load("Net7_MS_fold4_Test.npz")
#Net7_f5_Val, Net7_f5_Test = np.load("Net7_MS_fold5_Val.npz"),np.load("Net7_MS_fold5_Test.npz")

Net8_f1_Val, Net8_f1_Test = np.load("Net8_MS_cat_fold1_Val.npz"),np.load("Net8_MS_cat_fold1_Test.npz")
Net8_f2_Val, Net8_f2_Test = np.load("Net8_MS_cat_fold2_Val.npz"),np.load("Net8_MS_cat_fold2_Test.npz")
Net8_f3_Val, Net8_f3_Test = np.load("Net8_MS_cat_fold3_Val.npz"),np.load("Net8_MS_cat_fold3_Test.npz")
Net8_f4_Val, Net8_f4_Test = np.load("Net8_MS_cat_fold4_Val.npz"),np.load("Net8_MS_cat_fold4_Test.npz")
Net8_f5_Val, Net8_f5_Test = np.load("Net8_MS_cat_fold5_Val.npz"),np.load("Net8_MS_cat_fold5_Test.npz")

#Merge List by Nets/Folds
Net1_folds_val  = [Net1_f1_Val, Net1_f2_Val, Net1_f3_Val, Net1_f4_Val, Net1_f5_Val]
Net1_folds_test = [Net1_f1_Test, Net1_f2_Test, Net1_f3_Test, Net1_f4_Test, Net1_f5_Test]
Net5_folds_val  = [Net5_f1_Val, Net5_f2_Val, Net5_f3_Val, Net5_f4_Val, Net5_f5_Val]
Net5_folds_test = [Net5_f1_Test, Net5_f2_Test, Net5_f3_Test, Net5_f4_Test, Net5_f5_Test]
Net8_folds_val  = [Net8_f1_Val, Net8_f2_Val, Net8_f3_Val, Net8_f4_Val, Net8_f5_Val]
Net8_folds_test = [Net8_f1_Test, Net8_f2_Test, Net8_f3_Test, Net8_f4_Test, Net8_f5_Test]


def AppendList(List):
    ResList = []
    for i in List:
        ResList += list(i)
    return np.array(ResList)

#def MergeNpzs(NETS_PATH):
#    if op.isfile(Data_Path):
#        return
#    X,targets,predicts = [],Npzs[0]['targets'],[]
    
#        print('{}\t{}'.format(targets[i],pred[0][0].cpu().detach().numpy()))
#        print('{}\t{}'.format(targets[i],CLASSES[int(pred[0][0].cpu().detach().numpy())]))
#    np.savez(CONCAT_DATA_PATH, X=np.array(X), targets=np.array(targets), predicts=np.array(predicts))

def ConcatFoldsVal(NET_VAL_NPZ_PATH,Net_folds_val):
    Net_f1,Net_f2,Net_f3,Net_f4,Net_f5 = Net_folds_val
    X = AppendList([Net_f5['X'],Net_f4['X'],Net_f3['X'],Net_f2['X'],Net_f1['X']])
    X = X.reshape([-1,9])
    Predicts = AppendList([Net_f5['predicts'],Net_f4['predicts'],Net_f3['predicts'],Net_f2['predicts'],Net_f1['predicts']])
    Targets  = AppendList([Net_f5['targets'],Net_f4['targets'],Net_f3['targets'],Net_f2['targets'],Net_f1['targets']])
    np.savez(NET_VAL_NPZ_PATH, X=np.array(X), targets=np.array(Targets), predicts=np.array(Predicts))
    return np.load(NET_VAL_NPZ_PATH)

def AvgFoldsTest(NET_TEST_NPZ_PATH,Net_folds_test):
#def AvgFoldsTest(Net_folds_test):
    Net_folds_X = [Npz['X'] for Npz in Net_folds_test]
    X = []
    for i in tqdm(range(len(Net_folds_X[0]))):
        X_Merge = []
        for Net_fold_X in Net_folds_X:
            X_Merge += list(Net_fold_X[i])
        preds = torch.zeros(len(CLASSES))
        for j in range(len(X_Merge)):
            preds = preds+torch.nn.functional.normalize(torch.from_numpy(np.array([X_Merge[j]])))
        preds = preds.cpu().detach().numpy()
        X.append(preds/len(Net_folds_X))
    X = np.array(X).reshape([-1,9])
    Predicts = [np.argmax(line) for line in X]
    Targets  = Net_folds_test[0]['targets']
    np.savez(NET_TEST_NPZ_PATH, X=np.array(X), targets=np.array(Targets), predicts=np.array(Predicts))
    return np.load(NET_TEST_NPZ_PATH)


Net1_Concat_Val = ConcatFoldsVal('Net1_Concat_Val.npz',Net1_folds_val)
Net5_Concat_Val = ConcatFoldsVal('Net5_Concat_Val.npz',Net5_folds_val)
Net8_Concat_Val = ConcatFoldsVal('Net8_Concat_Val.npz',Net8_folds_val)
Net1_Avg_Test = AvgFoldsTest('Net1_Avg_Test.npz',Net1_folds_test)
Net5_Avg_Test = AvgFoldsTest('Net5_Avg_Test.npz',Net5_folds_test)
Net8_Avg_Test = AvgFoldsTest('Net8_Avg_Test.npz',Net8_folds_test)

