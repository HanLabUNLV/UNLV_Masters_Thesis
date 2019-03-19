
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
import itertools
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd
import datetime
from matplotlib import axes
import matplotlib.gridspec as gridspec

import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.decomposition import PCA
from concurrent.futures import ProcessPoolExecutor,wait
from concurrent import futures
import traceback
import multiprocessing as mp
from multiprocessing import Manager
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import time
from sklearn.utils.multiclass import unique_labels



def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


#np.set_printoptions(precision=2)



'''
#selectKBest
svm=pd.read_csv('/media/sravani/Data/UNLV/Thesis/version2/SelectKBest_knn_impute/SVM_labels_predicted.csv',sep=',')
rf=pd.read_csv('/media/sravani/Data/UNLV/Thesis/version2/SelectKBest_knn_impute/random_forest_labels_predicted.csv',sep=',')
dt=pd.read_csv('/media/sravani/Data/UNLV/Thesis/version2/SelectKBest_knn_impute/decision_trees_classifier_labels_predicted.csv',sep=',')
knn=pd.read_csv('/media/sravani/Data/UNLV/Thesis/version2/SelectKBest_knn_impute/KNN_Classfier_labels_predicted.csv',sep=',')
todo=[svm,dt,rf,knn]
names=['SVM','Decision Trees','Random Forest','KNN']
'''
'''
svm=pd.read_csv('/media/sravani/Data/UNLV/Thesis/version2/PCA_mean_impute/PCA_SVM_labels_predicted.csv',sep=',')
rf=pd.read_csv('/media/sravani/Data/UNLV/Thesis/version2/PCA_mean_impute/PCA_random_forest_labels_predicted.csv',sep=',')
dt=pd.read_csv('/media/sravani/Data/UNLV/Thesis/version2/PCA_mean_impute/PCA_decision_trees_labels_predicted.csv',sep=',')
todo=[svm,dt,rf]
names=['SVM','Decision Trees','Random Forest']
'''
svm=pd.read_csv('/media/sravani/Data/UNLV/Thesis/version2/PCA_knn_impute_k1/PCA_SVM_labels_predicted.csv',sep=',')
rf=pd.read_csv('/media/sravani/Data/UNLV/Thesis/version2/PCA_knn_impute_k1/PCA_random_forest_labels_predicted.csv',sep=',')
dt=pd.read_csv('/media/sravani/Data/UNLV/Thesis/version2/PCA_knn_impute_k1/PCA_decision_trees_labels_predicted.csv',sep=',')
todo=[svm,dt,rf]
names=['SVM','Decision Trees','Random Forest']

classes=['BRCA','HNSC','THCA','PRAD',
         'LGG','LUAD','KIRC','UCEC',
         'SKCM','BLCA','LIHC','LUSC',
         'STAD','COAD','KIRP','CESC',
         'SARC','ESCA','PAAD','PCPG',
         'TGCT','GBM','LAML','THYM','READ'
         ]

for n in names:
    #temp=knn
    idx=names.index(n)
    temp=todo[idx]
    actual=np.array(temp['actual'])
    predicted=np.array(temp['predicted'])
    print(len(actual))
    print(len(predicted))
    print(confusion_matrix(actual,predicted))
    print(np.unique(temp['predicted']))
    # Plot non-normalized confusion matrix

    plot_confusion_matrix(actual,predicted, classes=classes,
                          title='Confusion Matrix - '+names[idx])


    plt.show()