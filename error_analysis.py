'''
idata_labels = pd.read_csv('/data2/han_lab/sravani/Data/IndependentData/independent_data_labels.csv', sep=',')
idata_labels = idata_labels.set_index('Sample ID')

predicted=pd.read_csv('/data2/han_lab/sravani/PCA/labels_predicted_k_1.csv',sep=',')
predicted=predicted.set_index('barcode')
predicted=predicted.drop(columns='Unnamed: 0')

x=idata_labels.iloc[idata_labels.index.isin(predicted.index)]

predicted=pd.concat([predicted,x],axis=1,sort=False)
predicted=predicted.reset_index()
predicted.to_csv('/data2/han_lab/sravani/PCA/labels_predicted_analysis.csv',sep=',',index=False)

import os
cmd = 'scp '+'/data2/han_lab/sravani/PCA/labels_predicted_analysis.csv'+' gannavar@bobby.cs.unlv.edu://home/gannavar/UNLV_Thesis/'
os.system(cmd)
'''

import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix

# import some data to play with

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


df=pd.read_csv('/media/sravani/Data/UNLV/Thesis/version2/Pipeline_results/labels_predicted.csv')
y_pred=df['predicted']
y=df['actual']
class_names=np.unique(y_pred)
# Compute confusion matrix
cnf_matrix = confusion_matrix(y, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

'''
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
'''
plt.show()
