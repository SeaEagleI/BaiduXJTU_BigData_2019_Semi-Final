import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

CLASSES = ['00{}'.format(i) for i in range(1,10)]

#This python file is to test a single network on dataset and visiualise confusion metrix.
def plot_confusion_matrix(cm, classes, ModelName, normalize=False,  cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
#    if normalize:
#    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#    else:
#        print('Confusion matrix, without normalization')
    plt.figure(figsize = (11,11))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    print(np.diag(cm))
    plt.title('{} After Normalization'.format(ModelName),size=30)
#    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label',size=20)
    plt.xlabel('Predicted label',size=20)
    plt.tight_layout()

def LoadMatrixFromNpz(npz_path):
    Net = np.load(npz_path)
    targets  = Net['targets']
    predicts = Net['predicts']
#    print('Accuracy of Orignal Input: %0.6f'%(accuracy_score(predicts, targets, normalize = True)))

    cnf_matrix = confusion_matrix(targets, predicts)
    cnf_tr = np.trace(cnf_matrix)
    cnf_tr = cnf_tr.astype('float')
#    plt.figure()
#    plot_confusion_matrix(cnf_matrix, classes = CLASSES ,title='Confusion matrix, without normalization')
#    plt.figure()
    MODEL_NAME = npz_path.split('/')[-1].replace('_',' ').replace('.npz','')
    plot_confusion_matrix(cnf_matrix,CLASSES,MODEL_NAME,normalize=True)
    plt.show()


LoadMatrixFromNpz('./Tensors/folds_concat/Net1-5_fold1_Val.npz')
LoadMatrixFromNpz('./Tensors/folds_concat/Nets(1-5-8)_Concat_Val.npz')

