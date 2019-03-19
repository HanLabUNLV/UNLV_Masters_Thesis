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
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from concurrent.futures import ProcessPoolExecutor, wait
from concurrent import futures
import traceback
import multiprocessing as mp
from multiprocessing import Manager
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import time
import os
os.environ['QT_QPA_PLATFORM']='offscreen'

global algorithm_pipeline
global algo_best_params



class ProcessManager:
    def __init__(self):
        self.processes = []

    def add_process(self, fn, chrm,
                    impute_method, scaling,
                    kNN_feature_selection, impute_k,
                    n_features, n_neighbours, output):
        self.processes.append((fn, chrm,
                               impute_method, scaling,
                               kNN_feature_selection, impute_k,
                               n_features, n_neighbours, output))

    def launch(self, max_workers=4):

        futs = []
        ex = ProcessPoolExecutor(max_workers=max_workers)
        for p in self.processes:
            try:
                fut = ex.submit(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8])
                futs.append(fut)
            except:
                print(traceback.print_exc())
        return futs


def get_confusion_matrix(actual, predicted):
    return confusion_matrix(actual, predicted)


def get_classification_report(actual, predicted):
    return classification_report(actual, predicted)


def get_weighted_precision_recall_fscore_support(actual, predicted):
    precision, recall, fscore, support = score(actual, predicted, average='weighted', labels=np.unique(actual))
    return precision, recall, fscore, support


def get_binarized_labels(y):
    n_clases = np.unique(y)
    y_bin = label_binarize(y, clases=[i for i in range(len(n_clases))])
    return y_bin, n_clases


def one_vs_rest_classifier(X_train, y_train, X_test, y_test, model):
    '''

    :param X_train:
    :param y_train: binarized labels
    :param X_test:
    :param y_test: binarized labels
    :return:
    '''
    classifier = OneVsRestClassifier(model)
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    return y_score


def get_roc_auc_score(y_test, y_score, n_classes, file_name):
    '''

    :param y: binarized labels
    :param y_score: result of model.decision_function()
    :return:
    '''
    # y_score you will get from decision_function

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(file_name)


def plot_mean(df, title, filename):
    ax = df.mean().plot(kind='hist', figsize=[12, 6], title=title)
    fig = ax.get_figure()
    fig.savefig(filename)


def plot_describe(df, title, filename):
    ax = df.describe().plot(kind='hist', figsize=[12, 6], title=title)
    fig = ax.get_figure()
    fig.savefig(filename)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    to use: plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

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


def get_train_data(chrm):
    # load the labels
    # get without NA data
    training_labels = pd.read_csv('/data2/han_lab/sravani/heterogeneous_dataset/heterogeneous_labels.csv', sep='\t')
    #training_labels = training_labels.set_index('Barcode')

    d = pd.read_csv('/data2/han_lab/richardvan/heterogenousDatasets/forMethylationPipeline/dataset_90percentTCGA/training/' + chrm + '_data.csv')
    d = d.set_index('Composite')
    dT = d.T  # transpose it
    dT=dT.dropna(axis='columns')
    print(chrm + " : TCGA withoutNA Shape:" + str(dT.shape))

    labels_req=training_labels[training_labels['Barcode'].isin(dT.index)]
    labels_req=labels_req.set_index('Barcode')
    # drop probes with mean <0.2
    #dT = dT[dT.columns[dT.mean(axis=0) > 0.2]]
    #print(chrm + " : TCGA after removing row means <0.2 Shape:" + str(dT.shape))

    # merge dfT and labels
    data = pd.concat([dT, labels_req], axis=1,sort=False)
    print(chrm + " : Training shape(after adding labels):" + str(data.shape))

    # remove samples with type count <100
    temp = data.groupby(['Label']).size().reset_index()
    filter = temp[temp[0] > 100]
    train_filter = data[data['Label'].isin(filter['Label'])]
    print(chrm + " : Train Shape (after removing samples with count(type) <100):" + str(train_filter.shape))

    # drop few uneccesary columns
    X = train_filter.drop(columns=['Label'])
    y = train_filter['Label']
    probes = [i for i in X.columns]

    return X, y


def get_independent_data(chrm, train_X, train_y):
    idata_labels = pd.read_csv('/data2/han_lab/sravani/heterogeneous_dataset/heterogeneous_labels.csv', sep='\t')
    idata_labels = idata_labels.set_index('Barcode')
    IData_path = '/data2/han_lab/richardvan/heterogenousDatasets/forMethylationPipeline/dataset_90percentTCGA/testing/'
    file = IData_path + chrm + '.csv'
    df = pd.read_csv(file, sep=',')
    df1 = df.drop(columns=['chr.' + str(i) for i in range(1, 21)])
    df1 = df1.drop(columns=['chr'])
    df2 = df1[df1[u'composite'].isin(train_X.columns)]

    df2 = df2.set_index([u'composite'])
    test = df2.T
    x = idata_labels[idata_labels.index.isin(test.index)]
    data = pd.concat([test, x], axis=1, sort=False)
    print (chrm + " : Independent data shape:" + str(data.shape))
    data = data.rename(columns={"TCGA Label ": "type"})
    req_labels = np.unique(train_y)
    data1 = data[data['type'].isin(req_labels)]
    print(chrm + " : Independent data shape (after removing labels not considered in training):" + str(data1.shape))
    '''
    #display rows with Null values
    df1 = data[data.isnull().any(axis=1)]
    #find if any lables are missing
    temp=data[data.isnull().any(axis=1)]
    null_index=temp[temp['TCGA Label '].isnull()].index.tolist()
    idata_labels.ix[null_index]  
    '''
    X = data1.drop(columns=['type', 'Tissue Type'])
    y = data1['type']
    idx = [i for i in train_X.columns]
    X = X[idx]

    return X, y


def plot_select_K_best(X, y, title, filename):
    model = SelectKBest(chi2, k='all')
    model.fit(X, y)
    X_new = model.transform(X)
    temp = pd.DataFrame(columns=['scores'])
    temp['scores'] = model.scores_
    temp.index = X.columns
    temp = temp.sort_values(by=['scores'], ascending=False)
    temp = temp.reset_index()
    ax = temp.plot(kind='hist', figsize=[12, 8], x='index', y='scores', title=title)
    leg = ax.get_legend()
    #####horizontal line
    horiz_line_data = np.array([100 for i in range(2000)])
    line1, = ax.plot([i for i in range(2000)], horiz_line_data, 'r--', label='100')
    line2, = ax.plot([i for i in range(2000)], np.array([50 for i in range(2000)]), 'b--', label='50')

    ax.legend((line1, line2), ('100', '50'))

    ax.grid(True, which='minor', axis='x')

    ax.grid(False, which='major', axis='x')
    fig = ax.get_figure()
    fig.savefig(filename)


def plots_training_featureScores(path):
    chromosomes = ['chr' + str(i) for i in range(1, 23)]
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    for chrm in chromosomes:
        X, y = get_train_data(chrm)
        # selectKbest plots
        filename = path + "training_" + chrm + "_SelectKBest_scores_" + timestamp + ".png"
        title = "Training: " + chrm + " \nSelectKBest Scores - Chi2 \n(total features:" + str(X.shape[1]) + ")"
        plot_select_K_best(X, y, title, filename)


def plot_mean_min_max(X, y, title, filename):
    temp = pd.DataFrame()
    temp['mean'] = X.mean()
    temp['25_q'] = X.quantile(q=0.25)
    temp['50_q'] = X.quantile(q=0.5)
    temp['75_q'] = X.quantile(q=0.75)
    temp = temp.sort_values(by='mean')
    temp = temp.reset_index().drop(columns='index')
    ax = temp.plot(kind='line', figsize=[12, 8], title=title)
    fig = ax.get_figure()
    fig.savefig(filename)


def get_k_best_features(X, y, k=20):
    model = SelectKBest(chi2, k=k)
    model.fit(X, y)
    X_new1 = model.transform(X)

    mask = model.get_support()  # list of booleans
    new_features = []  # The list of your K best features

    for bool, feature in zip(mask, X.columns):
        if bool:
            new_features.append(feature)

    X_new = pd.DataFrame(X_new1, columns=new_features)
    X_new.index = X.index
    t = X[X_new.columns]
    if X_new.equals(t):
        return X_new, y
    else:
        return


def plots_training_features(path):
    chromosomes = ['chr' + str(i) for i in range(1, 23)]
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    for chrm in chromosomes:
        X, y = get_train_data(chrm)
        # selectKbest plots
        filename = path + "training_" + chrm + "_raw_data_" + timestamp + ".png"
        title = "Training: " + chrm + " \n(total features:" + str(X.shape[1]) + ")"
        plot_mean_min_max(X, y, title, filename)


def perform_standard_scalling(X):
    # perform standard scaling
    scaler = StandardScaler()
    scaler.fit(X)
    X1 = scaler.transform(X)
    X1 = pd.DataFrame(X1, columns=X.columns)
    return X1


def get_nearest_neighbours(train, test, k=20):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(train)
    # print(nbrs)
    distances, indices = nbrs.kneighbors(test)
    return indices


def impute_missing_data(msg, indices, train, test):
    print("=====================================================================")
    print(msg + " Imputing...")
    print(msg + ": Recieved Train shape: " + str(train.shape) + " Test shape:" + str(test.shape))
    # get same cols in test and train
    # print(test.iloc[0:5,0:5])
    testT = test.T
    testT = testT.reset_index()
    # print(testT.iloc[0:5,0:5])
    test2 = testT[testT[u'index'].isin(train.columns)]
    test2 = test2.set_index('index')
    test2 = test2.T
    # test2.to_csv('/data2/han_lab/sravani/before_impute.csv', sep=',')
    print(msg + ": With Required Train shape: " + str(train.shape) + " Test shape:" + str(test2.shape))
    train = train.reset_index()
    test_impute = pd.DataFrame(columns=test2.columns)
    for i in range(indices.shape[0]):
        # print("=====================================")
        print(test2.index[i])
        sum_missing = test2.iloc[i, :].isnull().values.sum()
        # print(msg+": Index"+str(test2.index[i])+" - Number of missing in test:"+str(test2.iloc[i, :].isnull().values.sum()))
        if sum_missing > 0:
            rows = list(indices[i])
            temp = train.loc[rows, :]
            # temp = temp.set_index('Barcode1')
            # print(temp)

            col_means = temp.mean()
            # print(col_means[u'cg22934785'])
            # temp_means = pd.DataFrame(0, index=np.arange(len(temp)), columns=temp.columns)
            # for col in temp_means.columns:
            #    temp_means[col]=train[col].mean()

            # test2[test2.isnull()] = temp_means

            x = test2.iloc[i, :]
            # x.to_csv('/data2/han_lab/sravani/impute1.csv',sep=',')
            # print(chrm+" :imputing..."+str(test2.index[i]))
            x[x.isnull()] = col_means
            # x.to_csv('/data2/han_lab/sravani/impute2.csv', sep=',')

            # print(chrm+": Number of missing in test after imputing:"+str(x.isnull().values.sum()))
            test_impute = test_impute.append(x)
            # print(chrm+": Impute size:"+str(test_impute.shape))

        else:
            test_impute = test_impute.append(test2.iloc[i, :])
    # print(chrm+': writing to file: /data2/han_lab/sravani/Data/IndependentData/imputed/'+chrm+'_imputed_test.csv')
    # test_impute.to_csv('/data2/han_lab/sravani/Data/IndependentData/imputed/'+chrm+'_imputed.csv', sep=',')
    return test_impute


def perform_kNN_impute(msg, train_X, train_y,
                       test_X, test_y,
                       n_features_impute, n_neighbours):
    # remove missing values from test
    if test_X.isnull().values.any():
        test_withoutNA = test_X.dropna(axis='columns')
        print(msg + ": Test Shape (after removing missing):" + str(test_withoutNA.shape))
    else:
        test_withoutNA = test_X
    # extract common columns in test and production
    trainT = train_X.T
    x = trainT[trainT.index.isin(test_withoutNA.columns)]
    train_comm_X = x.T  # has common columns

    print("\n" + msg + ": Train Shape (only with common columns):" + str(train_comm_X.shape))

    testT = test_withoutNA.T
    test_comm_X = testT[testT.index.isin(train_comm_X.columns)].T  # has common columns
    print(msg + ": Test shape (only with common columns):" + str(test_comm_X.shape))

    # get K best features from train

    print(msg + " : Selecting k best features in Training...")
    train_X_best, train_y = get_k_best_features(train_comm_X, train_y, k=n_features_impute)
    print(msg + " : After selecting k best features in Training shape:" + str(train_X_best.shape))

    test_X_best = test_comm_X[train_X_best.columns]
    print(msg + " : After selecting k best features in test shape:" + str(test_X_best.shape))

    # get K nearest neighbours
    print("\n" + msg + ": Imputing - Applying nearest neighbours...")
    indices = get_nearest_neighbours(train_X_best, test_X_best, k=n_neighbours)
    # indices = get_nearest_neighbours(train_X_best, test_X_best, k=1)

    print(msg + ": Imputing - Indices shape:" + str(indices.shape))
    test_impute = impute_missing_data(msg, indices, train_X, test_X)
    print(msg + " : Impute finished..." + str(test_impute.shape))
    print(" idata_X_impute:", test_impute.iloc[0:5, 0:5])
    return test_impute


def perform_over_sampling_with_cleaning(X, y):
    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_sample(X, y)
    return X_res, y_res


def perform_mean_impute(df):
    if df.isnull().values.any():
        # fill missing values with column means( probe mean)
        return df.fillna(df.mean())


def do_task_for_each_chromosome(chrm, impute_method,
                                scaling, kNN_feature_selection,
                                n_features, n_features_impute,
                                n_neighbours, output):
    # get train data
    print(chrm + " : Getting TCGA Data...")
    train_X, train_y = get_train_data(chrm)
    print(chrm + " : Getting independent Data...")
    idata_X, idata_y = get_independent_data(chrm, train_X, train_y)

    best_results = {}

    if idata_X.isnull().values.any():
        if impute_method == 'kNN':
            print(chrm + " : Performing kNN impute..")
            idata_X_impute = perform_kNN_impute(chrm, train_X, train_y,
                                                idata_X, idata_y,
                                                n_features_impute, n_neighbours)
            print(chrm + " : kNN impute finsihed ! Shape:" + str(idata_X_impute.shape))
        if impute_method == 'mean':
            print(chrm + " : Performing mean impute..")

            idata_X_impute = perform_mean_impute(idata_X)

        # give index for idat_X_impute
        print("index of idata_X_impute:", idata_X_impute.index)

        print(chrm + " : Output size (before):" + str(len(output)))

        print(chrm + " : Selecting k best features in Training...")
        train_X_new, train_y = get_k_best_features(train_X, train_y, k=n_features)
        print(chrm + " : After selecting k best features in Training shape:" + str(train_X_new.shape))

        print(chrm + " : Selecting k best features in Independent...")
        idata_X_new = idata_X_impute[train_X_new.columns]
        print(chrm + " : After selecting k best features in independent shape:" + str(idata_X_new.shape))

        output[chrm] = {'training': train_X_new,
                        't_labels': train_y,
                        'independent': idata_X_new,
                        'i_labels': idata_y,
                        }
        print(chrm + " : Output size (after):" + str(len(output)))
    else:
        print(chrm + " : Number of probes mismatch in Independent Data and train data.")


def do_task_for_all_chromosomes(tcga_filename, idata_filename, impute_method, scaling,
                                kNN_feature_selection, n_features,
                                n_features_impute, n_neighbours):
    '''
    missing value imputation with KNN data
    :return:
    '''
    # if __name__==__main__():
    chromosomes = ['chr' + str(i) for i in range(22, 23)]

    merge_tdata = None  # holds PCA on training data for 22 chromosomes
    merge_idata = None  # holds PCA on independent data for 22 chromosomes

    max_workers = 10
    pm = ProcessManager()
    m = Manager()
    output = m.dict()
    for chrm in chromosomes:
        pm.add_process(do_task_for_each_chromosome, chrm,
                       impute_method, scaling,
                       kNN_feature_selection, n_features, n_features_impute, n_neighbours, output)

    futs = pm.launch(max_workers=max_workers)
    print(wait(futs, return_when=futures.ALL_COMPLETED))
    print("Output:", len(output))
    for key, val in output.items():
        if merge_tdata is None:
            merge_tdata = val['training']
            print(key + " : training - After merge:" + str(merge_tdata.shape))
            merge_tdata = pd.concat([merge_tdata, val['t_labels']], axis=1)
            print(key + " : training - After adding labels:" + str(merge_tdata.shape))
        else:
            print(key + " : training - Merging with other chromosomes..")
            merge_tdata = pd.concat([merge_tdata, val['training']], axis=1)  # merge with other chromosomes
            print(key + " : training - After merge:" + str(merge_tdata.shape))
        if merge_idata is None:
            merge_idata = val['independent']
            print(key + " : Independent - After merge:" + str(merge_idata.shape))
            merge_idata = pd.concat([merge_idata, val['i_labels']], axis=1)
            print(key + " : Independent - After adding labels:" + str(merge_idata.shape))
        else:
            print(key + " : Independent - Merging with other chromosomes..")
            merge_idata = pd.concat([merge_idata, val['independent']], axis=1)  # merge with other chromosomes
            print(key + " : Independent - After merge:" + str(merge_idata.shape))

    print("Process complete")
    # print("Writing merged data to file...")
    merge_tdata.to_csv(tcga_filename, sep=',')
    merge_idata.to_csv(idata_filename, sep=',')
    return merge_tdata, merge_idata


def load_data_from_file(tcga_filename, idata_filename):
    tcga = pd.read_csv(tcga_filename, sep=',')
    idata = pd.read_csv(idata_filename, sep=',')
    return tcga, idata


def process(algorithm, load_from_file,
            tcga_filename, idata_filename,
            impute_method,
            scaling, kNN_feature_selection,
            n_features, n_features_impute,
            n_neighbours, run_pipeline=False):
    if load_from_file:
        print("Extracting TCGA and independent data from file..")

        tcga, idata = load_data_from_file(tcga_filename, idata_filename)
        tcga = tcga.set_index('Barcode1')
        idata = idata.set_index('Unnamed: 0')
        # print(tcga.iloc[0:5,0:5])
        # print(idata.iloc[0:5,0:5])

    else:
        print("Creating TCGA and independent data ...")
        tcga, idata = do_task_for_all_chromosomes(tcga_filename, idata_filename, impute_method, scaling,
                                                  kNN_feature_selection, n_features,
                                                  n_features_impute, n_neighbours)
        print("Data Extraction complete!")
    train_X = tcga.drop(columns=['type'])
    train_y = tcga['type']

    idata_X = idata.drop(columns=['type'])
    idata_y = idata['type']

    best_results = {'algorithm': algorithm['name']}

    if run_pipeline:
        print("Running pipeline....")
        gs = GridSearchCV(estimator=algorithm['pipeline'],
                          param_grid=algorithm['grid_params'],
                          scoring='accuracy',
                          cv=10,
                          n_jobs=-1,
                          verbose=10)
        X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.2, random_state=42)

        gs.fit(X_train, y_train)
        # Best params
        print('Best params: %s' % gs.best_params_)
        best_results['best_params'] = gs.best_params_
        # Best training data accuracy
        print('Best training accuracy: %.3f' % gs.best_score_)
        best_results['best_training_accuracy'] = gs.best_score_
        # Predict on test data with best params
        y_pred = gs.predict(X_test)
        precision, recall, fscore, support = get_weighted_precision_recall_fscore_support(y_test, y_pred)

        # Test data accuracy of model with best params
        print('Test set accuracy score for best params: %.3f ' % accuracy_score(y_test, y_pred))
        best_results['test_accuracy'] = accuracy_score(y_test, y_pred)

        print("\nWriting results to file...")
        with open('/data2/han_lab/sravani/heterogeneous_dataset/pipeline/' + algorithm['name'] + '_pipeline_results.txt', 'w') as f:
            f.write('\n\n\nEstimator: ' + str(algorithm['name']) +
                    '\nBest params: ' + str(gs.best_params_) +
                    '\nBest training accuracy: ' + str(gs.best_score_) +
                    '\nTest set accuracy score for best params: ' + str(accuracy_score(y_test, y_pred)) +
                    '\nTest precision:' + str(precision) +
                    '\nTest recall:' + str(recall) +
                    '\nTest fscore:' + str(fscore) +
                    '\nTest support:' + str(support))

        best_results['test_precision'] = precision
        best_results['test_recall'] = recall
        best_results['test_fscore'] = fscore
        best_results['test_support'] = support
        print("Best Results:", best_results)
        best_model = gs.best_estimator_

    else:
        print('Getting best model...')
        best_model = algorithm['best']

    print("Fitting best model with train data...")
    best_model.fit(train_X, train_y)
    print("Predicting independent data on trained model...")
    idata_y_pred = best_model.predict(idata_X)

    i_precision, i_recall, i_fscore, i_support = get_weighted_precision_recall_fscore_support(idata_y, idata_y_pred)
    best_results['idata_accuracy'] = accuracy_score(idata_y, idata_y_pred)
    best_results['idata_precision'] = i_precision
    best_results['idata_recall'] = i_recall
    best_results['idata_fscore'] = i_fscore
    best_results['idata_support'] = i_support
    print("\nWriting results to file...")
    with open('/data2/han_lab/sravani/heterogeneous_dataset/pipeline/' + algorithm['name'] + '_pipeline_results.txt', 'a') as f:
        f.write('\n\n\n\nIndependent accuracy:' + str(accuracy_score(idata_y, idata_y_pred)) +
                '\nIndependent precision:' + str(i_precision) +
                '\nIndependent recall:' + str(i_recall) +
                '\nIndependent fscore:' + str(i_fscore) +
                '\nIndependent support:' + str(i_support)
                )
    print(best_results)

    # print(idata_y.index)
    temp = pd.DataFrame()
    temp['actual'] = idata_y
    temp['predicted'] = idata_y_pred
    temp.index = idata.index
    temp.to_csv('/data2/sravani/heterogeneous_dataset/version2/pipeline/labels_predicted.csv')
    # print(temp.iloc[0:5,0:5])
    return best_results


#####################################################
# plot raw data
# plots_training_features('/data2/han_lab/sravani/version2/')

# to plot feature score according SklearnKBest
# plots_training_featureScores('/data2/han_lab/sravani/version2/')

def combine_labels_info():
    idata_labels = pd.read_csv('/data2/han_lab/sravani/Data/IndependentData/independent_data_labels.csv', sep=',')
    training_labels = pd.read_csv('/data2/han_lab/sravani/Data/patientToCancerLabels_final.csv', sep=',')
    temp=pd.DataFrame()
    temp1=training_labels[[u'Barcode1',u'type']]
    temp2=idata_labels[[u'Sample ID',u'TCGA Label ']]
    temp1=temp1.rename(columns={'Barcode1':'Barcode','type':'Label'})
    temp2=temp2.rename(columns={'TCGA Label ':'Label','Sample ID':'Barcode'})
    merge=temp1.append(temp2,sort=False)
    merge.to_csv('/data2/han_lab/sravani/heterogeneous_dataset/heterogeneous_labels.csv',sep='\t',index=False)


t1=time.time()
scaling = [True, False]
impute_method = ['kNN', 'mean']
kNN_feature_selection = ['PCA', 'SelectKBest']
algorithm_pipeline = {'decision_trees': {'name': 'decision_trees_classifier',
                                         'pipeline': Pipeline([('clf', DecisionTreeClassifier(random_state=42))]),
                                         'grid_params': [{'clf__criterion': ['gini', 'entropy'],
                                                          'clf__min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                                          'clf__max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                                          'clf__min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10]}],
                                         'best': DecisionTreeClassifier(criterion='entropy', max_depth=9,
                                                                        min_samples_leaf=6, min_samples_split=2,
                                                                        random_state=42)

                                         },
                      'random_forest': {'name': 'random_forest',
                                        'pipeline': Pipeline([('clf', RandomForestClassifier(random_state=42))]),
                                        'grid_params': [{'clf__criterion': ['gini', 'entropy'],
                                                         'clf__min_samples_leaf': [5, 6, 7, 8, 9, 10],
                                                         'clf__max_depth': [5, 6, 7, 8, 9, 10],
                                                         'clf__min_samples_split': [2, 3, 4, 5]}],
                                        'best': {RandomForestClassifier(random_state=42, criterion='entropy',
                                                                        max_depth=10, min_samples_leaf=9,
                                                                        min_samples_split=2)}

                                        },

                      'SVM': {'name': 'SVM',
                              'pipeline': Pipeline([('clf', svm.SVC(random_state=42))]),
                              'grid_params': [{'clf__kernel': ['linear'],
                                               'clf__C': [1, 2]}],
                              'best': svm.SVC(C=1, kernel='linear', random_state=42)

                              },
                      'kNN_classifier': {'name': 'KNN_Classfier',
                                         'pipeline': Pipeline([('clf', KNeighborsClassifier())]),
                                         'grid_params': [{'clf__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                                          'clf__weights': ['uniform', 'distance']}],
                                         'best': KNeighborsClassifier(n_neighbors=4, weights='distance')

                                         },

                      # 'naive_bayes':{'name':'naive_bayes',
                      #               'pipeline':Pipeline([('clf', GaussianNB())]),
                      #                'grid_params':[{'clf__var_smoothing':[0.000000001,0.00000001]}]}
                      }

n_features = 50  # number of features to consider in TCGA for model selction
n_features_impute = 20  # number of features to fit kNN for purspose of imputing
n_neighbours = 5  # number of neighbours to consider to take mean of hose values and fill in
kNN_feature_selection = 'SelectKBest'  # ['PCA','SelectKBest']
load_from_file = False  # True: loads daat from file, False: build the file ffrom scratch
scaling = False  # True: does standard scaling #False: Doesnot scale
run_pipeline = False
impute_method = 'kNN'

tcga_filename = '/data2/han_lab/sravani/heterogeneous_dataset/Heter_TCGA_' + kNN_feature_selection + '_k' + str(
    n_features) + '.csv'
idata_filename = '/data2/han_lab/sravani/heterogeneous_dataset/Heter_Idata_' + impute_method + '_impute_k' + str(
    n_features_impute) + '_neigh_' + str(n_neighbours) + '.csv'

# to run single algo

res = process(algorithm_pipeline['SVM'], load_from_file,
              tcga_filename, idata_filename, impute_method,
              scaling, kNN_feature_selection,
              n_features, n_features_impute,
              n_neighbours, run_pipeline)
# uncomment below to run all algos
'''
res={}
for key,val in algorithm_pipeline.items():
    print(key)
    if (key!='decision_trees') and (key!='random_forest'):
        res[key]   =process(algorithm_pipeline[key],load_from_file,
            tcga_filename, idata_filename,impute_method,
                scaling,kNN_feature_selection,
                n_features,n_features_impute,
                n_neighbours,run_pipeline)

'''
print(res)
print(time.time()-t1)

