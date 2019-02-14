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
import time
import os

global flag_standard_scaling
global flag_pipeline_to_seclect_model

class ProcessManager:
    def __init__(self):
        self.processes = []

    def add_process(self, fn, args,k,output):
        self.processes.append((fn,args,k,output))

    def launch(self, max_workers=4):

        futs = []
        ex = ProcessPoolExecutor(max_workers=max_workers)
        for p in self.processes:
            try:
                fut = ex.submit(p[0], p[1],p[2],p[3])
                futs.append(fut)
            except:
                print(traceback.print_exc())
        return futs

def get_train_data(chrm,training_labels):
    # load the labels
    #get without NA data

    d = pd.read_csv('/data2/han_lab/sravani/Data/without_NA/' + chrm + '_withoutNA.csv')
    d = d.set_index('Composite')
    dT = d.T  # transpose it
    print(chrm+" : TCGA withoutNA Shape:" + str(dT.shape))

    # drop probes with mean <0.2
    dT = dT[dT.columns[dT.mean(axis=0) > 0.2]]
    print(chrm+" : TCGA after removing row means <0.2 Shape:" + str(dT.shape))


    # merge dfT and labels
    data = pd.concat([dT, training_labels], axis=1)
    print(chrm+" : Training shape(after adding labels):"+ str(data.shape))

    # remove samples with type count <100
    temp = data.groupby(['type']).size().reset_index()
    filter = temp[temp[0] > 100]
    train_filter = data[data['type'].isin(filter['type'])]
    print(chrm + " : Train Shape (after removing samples with count(type) <100):"+ str(train_filter.shape))

    # drop few uneccesary columns
    X = train_filter.reset_index().drop(columns=['Barcode', 'Label', 'Barcode1', 'TSS', 'Center', 'Plate', 'type'])
    y = train_filter.reset_index()['type']
    probes=[i for i in X.columns]

    return X,y,probes,list(filter['type'])

def get_independent_data(chrm,idata_labels,probes,req_labels):
    IData_path = '/data2/han_lab/sravani/Data/IndependentData/per_chromosome/'
    file = IData_path + chrm + '.csv'
    df = pd.read_csv(file, sep=',')
    df1 = df.drop(columns=['chr.' + str(i) for i in range(1, 21)])
    df1 = df1.drop(columns=['chr'])
    df2 = df1[df1[u'composite'].isin(probes)]

    df2 = df2.set_index([u'composite'])
    test = df2.T
    x=idata_labels[idata_labels.index.isin(test.index)]
    data = pd.concat([test, x], axis=1,sort=False)
    print (chrm+" : Independent data shape:"+ str(data.shape))
    data = data.rename(columns={"TCGA Label ": "type"})
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
    X = data1.reset_index().drop(columns=['type', 'Tissue Type', 'index'])
    y = data1.reset_index()['type']
    idx=[i for i in probes]
    X=X[idx]
    barcodes = data1.reset_index().rename(columns={'index': 'barcode'})

    return X,y,barcodes['barcode']

def perform_standard_scalling(X):
    # perform standard scaling
    scaler = StandardScaler()
    scaler.fit(X)
    X1 = scaler.transform(X)
    X1 = pd.DataFrame(X1,columns=X.columns)
    return X1

def fill_missing_mean(df):
    if df.isnull().values.any():

        #fill missing values with column means( probe mean)
        return df.fillna(df.mean())

def impute_missing_data(chrm,indices, train, test):

    print("=====================================================================")
    print(chrm+" Imputing...")
    print(chrm+": Recieved Train shape: " + str(train.shape) + " Test shape:" + str(test.shape))
    # get same cols in test and train
    # print(test.iloc[0:5,0:5])
    testT = test.T
    testT = testT.reset_index()
    # print(testT.iloc[0:5,0:5])
    test2 = testT[testT[u'index'].isin(train.columns)]
    test2 = test2.set_index('index')
    test2 = test2.T
    #test2.to_csv('/data2/han_lab/sravani/before_impute.csv', sep=',')
    print(chrm+": With Required Train shape: " + str(train.shape) + " Test shape:" + str(test2.shape))
    train = train.reset_index()
    impute = pd.DataFrame(columns=test2.columns)
    for i in range(indices.shape[0]):
        #print("=====================================")
        #print(test2.index[i])
        sum_missing = test2.iloc[i, :].isnull().values.sum()
        #print(chrm+": Index"+str(test2.index[i])+" - Number of missing in test:"+str(test2.iloc[i, :].isnull().values.sum()))
        if sum_missing > 0:
            rows = list(indices[i])
            temp = train.loc[rows, :]
            #temp = temp.set_index('Barcode1')
            # print(temp)

            col_means = temp.mean()
            # print(col_means[u'cg22934785'])
            # temp_means = pd.DataFrame(0, index=np.arange(len(temp)), columns=temp.columns)
            # for col in temp_means.columns:
            #    temp_means[col]=train[col].mean()

            # test2[test2.isnull()] = temp_means

            x = test2.iloc[i, :]
            # x.to_csv('/data2/han_lab/sravani/impute1.csv',sep=',')
            #print(chrm+" :imputing..."+str(test2.index[i]))
            x[x.isnull()] = col_means
            # x.to_csv('/data2/han_lab/sravani/impute2.csv', sep=',')

            #print(chrm+": Number of missing in test after imputing:"+str(x.isnull().values.sum()))
            impute = impute.append(x)
            #print(chrm+": Impute size:"+str(impute.shape))

        else:
            impute = impute.append(test2.iloc[i, :])
    #print(chrm+': writing to file: /data2/han_lab/sravani/Data/IndependentData/imputed/'+chrm+'_imputed_test.csv')
    #impute.to_csv('/data2/han_lab/sravani/Data/IndependentData/imputed/'+chrm+'_imputed.csv', sep=',')
    return test2,impute

def get_nearest_neighbours(chrm,k,train_X,test_X):
    # remove missing values from test
    test_withoutNA = test_X.dropna(axis='columns')
    print(chrm + ": Test Shape (after removing missing):" +str(test_withoutNA.shape))

    # extract common columns in test and production
    trainT = train_X.T
    x = trainT[trainT.index.isin(test_withoutNA.columns)]
    train_comm_X = x.T  # has common columns

    print("\n" + chrm + ": Train Shape (only with common columns):"+str(train_comm_X.shape))

    testT = test_withoutNA.T
    test_comm_X = testT[testT.index.isin(train_comm_X.columns)].T  # has common columns
    print(chrm + ": Test shape (only with common columns):"+str(test_comm_X.shape))


    # apply PCA
    print("\n" + chrm + ": Imputing - Applying PCA on Train....")
    model = PCA(n_components=100)
    model.fit(train_comm_X)
    train_X_trans = model.transform(train_comm_X)
    print(chrm + ": Imputing - Train PCA shape:"+str(train_X_trans.shape))

    print("\n" + chrm + ": Imputing - Applying PCA on Test...")
    #model1 = PCA(n_components=100)
    #model1.fit(test_comm_X)
    test_X_trans = model.transform(test_comm_X)
    print(chrm + ": Imputing - Test PCA shape:"+str(test_X_trans.shape))

    # apply KNN
    print("\n" + chrm + ": Imputing - Applying nearest neighbours...")
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(train_X_trans)
    #print(nbrs)
    distances, indices = nbrs.kneighbors(test_X_trans)
    print(indices.shape)
    print(chrm + ": Imputing - Indices shape:" +str(indices.shape))

    return indices

def do_PCA_subtask_with_mean_impute(chrm, k,output):
    print("===========================================================================")
    print("chromosome:", chrm)
    print(chrm + " : loading training labels...")
    training_labels = pd.read_csv('/data2/han_lab/sravani/Data/patientToCancerLabels_final.csv', sep=',')
    training_labels = training_labels.set_index('Barcode1')

    # probes : those probes considered for PCA
    print(chrm + " : Getting training data...")
    train_X, train_y, probes ,req_labels= get_train_data(chrm, training_labels)

    print("")


    print(chrm + " : Loading independent data labels..")
    idata_labels = pd.read_csv('/data2/han_lab/sravani/Data/IndependentData/independent_data_labels.csv', sep=',')
    idata_labels = idata_labels.set_index('Sample ID')

    print(chrm + " : Getting independent data..")
    idata_X,idata_y,barcodes = get_independent_data(chrm,idata_labels,probes,req_labels)
    '''

    print(chrm + " : Getting independent imputed data..")
    idata_X, idata_y = get_imputed_data(chrm)
    '''
    if idata_X.shape[1] == train_X.shape[1]:
        if idata_X.isnull().values.any():
            print(chrm + " : Independent Data has missing values..")
            print(chrm + " : Filling missing values with mean..")
            idata_X_new = fill_missing_mean(idata_X)
            print(chrm + " : Fitting training data to model..")

            model = PCA(n_components=100)
            model.fit(train_X)

            print("")
            print(chrm + " : Transforming TCGA data according to model..")
            train_X_trans = model.transform(train_X)
            train_X_trans = pd.DataFrame(train_X_trans)
            train_X_trans = train_X_trans.add_prefix(str(chrm + '_PC_'))

            print("")
            print(chrm + " : Transforming independent data according to model..")
            idata_X_trans = model.transform(idata_X_new)
            idata_X_trans = pd.DataFrame(idata_X_trans)  # convert numpy array to dataframe
            idata_X_trans = idata_X_trans.add_prefix(str(chrm + '_'))  # give column names

            print(chrm + " : Output size (before):", len(output))
            output[chrm] = {'training': train_X_trans,
                            't_labels': train_y,
                            'independent': idata_X_trans,
                            'i_labels': idata_y,
                            'i_barcodes': barcodes}
            print(chrm + " : Data saved in shared variable")
            print(chrm + " : Output size (after):", len(output))
    else:
        print(chrm + " : Number of probes mismatch in Independent Data and train data.")

def do_PCA_subtask_with_kNN_impute(chrm,k,output):
    print(chrm + " : loading training labels...")
    training_labels = pd.read_csv('/data2/han_lab/sravani/Data/patientToCancerLabels_final.csv', sep=',')
    training_labels = training_labels.set_index('Barcode1')

    # probes : those probes considered for PCA
    print(chrm + " : Getting training data...")
    train_X, train_y, probes, req_labels = get_train_data(chrm, training_labels)

    if flag_standard_scaling:
        #perform standard scalling
        print(chrm+" : standard scalling on train..")
        train_X=perform_standard_scalling(train_X)

    print(chrm + " : Loading independent data labels..")
    idata_labels = pd.read_csv('/data2/han_lab/sravani/Data/IndependentData/independent_data_labels.csv', sep=',')
    idata_labels = idata_labels.set_index('Sample ID')

    print(chrm + " : Getting independent data..")
    idata_X, idata_y,barcodes = get_independent_data(chrm, idata_labels,probes, req_labels) #if probes= all , all probes are considered


    print(chrm+" : Finding nearest neighbours..")
    indices=get_nearest_neighbours(chrm,k,train_X,idata_X)
    #impute the missing values with the mean of neighbours
    test_req, test_imp_X = impute_missing_data(chrm, indices, train_X, idata_X)
    print(chrm+" : Missing values in imputed data:" + str(test_imp_X.isnull().values.sum()))
    print("\n"+chrm+" : Chromosome " + chrm + " imputation finished...!")

    if flag_standard_scaling:
    # perform standard scalling
        print(chrm + " : standard scalling on test..")
        test_imp_X= perform_standard_scalling(test_imp_X)

    if train_X.shape[1]==test_imp_X.shape[1]:
        print(chrm + " : Fitting training data to model..")

        model = PCA(n_components=100)
        model.fit(train_X)

        print(chrm + " : Transforming TCGA data according to model..")
        train_X_trans = model.transform(train_X)
        train_X_trans = pd.DataFrame(train_X_trans)
        train_X_trans = train_X_trans.add_prefix(str(chrm + '_PC_'))

        print("")
        print(chrm + " : Transforming independent data according to model..")
        idata_X_trans = model.transform(test_imp_X)
        idata_X_trans = pd.DataFrame(idata_X_trans)  # convert numpy array to dataframe
        idata_X_trans = idata_X_trans.add_prefix(str(chrm + '_'))  # give column names

        print(chrm + " : Output size (before):"+ str(len(output)))
        output[chrm] = {'training': train_X_trans,
                        't_labels': train_y,
                        'independent': idata_X_trans,
                        'i_labels': idata_y,
                        'i_barcodes':barcodes}
        print(chrm + " : Data saved in shared variable")
        print("===============================================================================")
        print(chrm + " : Output size (after):" + str(len(output)))

    else:
        print(chrm+" : Number of probes mismatch in Independent Data and train data.")

def create_PCA_data_v3():
    '''
    pca performed on training after removing labels with count(type)<100
    does mean imputtion for missing values
    does multiprocessing
    :return:
    '''
    #if __name__==__main__():
    chromosomes = ['chr'+str(i) for i in range(1,23)]

    merge_tdata=None #holds PCA on training data for 22 chromosomes
    merge_idata=None #holds PCA on independent data for 22 chromosomes

    max_workers = 10
    pm = ProcessManager()
    m = Manager()
    output=m.dict()
    for chrm in chromosomes:
        pm.add_process(do_PCA_subtask_with_mean_impute, chrm,k,output)

    futs=pm.launch(max_workers=max_workers)
    wait(futs, return_when=futures.ALL_COMPLETED)
    print("Output:",len(output))
    for key,val in output.items():
        if merge_tdata is None:
            merge_tdata = val['training']
            print(key+" : training - After merge:" + str(merge_tdata.shape))
            merge_tdata = pd.concat([merge_tdata, val['t_labels']], axis=1)
            print(key+" : training - After adding labels:" + str(merge_tdata.shape))
        else:
            print(key+" : training - Merging with other chromosomes..")
            merge_tdata = pd.concat([merge_tdata, val['training']], axis=1)  # merge with other chromosomes
            print(key+" : training - After merge:" + str(merge_tdata.shape))
        if merge_idata is None:
            merge_idata = val['independent']
            print(key+" : Independent - After merge:" + str(merge_idata.shape))
            merge_idata = pd.concat([merge_idata, val['i_labels']], axis=1)
            print(key+" : Independent - After adding labels:" + str(merge_idata.shape))
        else:
            print(key+" : Independent - Merging with other chromosomes..")
            merge_idata = pd.concat([merge_idata, val['independent']], axis=1)  # merge with other chromosomes
            print(key+" : Independent - After merge:" + str(merge_idata.shape))

    print("Process complete")
    #print("Writing merged data to file...")
    #merge_idata.to_csv('/data2/han_lab/sravani/PCA/IData_combined_PCA_data_v3.csv',index=False)
    #merge_tdata.to_csv('/data2/han_lab/sravani/PCA/TCGA_combined_PCA_data_v3.csv',index=False)

    return merge_tdata,merge_idata

def create_PCA_data_v4(k):
    '''
    missing value imputation with KNN data
    :return:
    '''
    #if __name__==__main__():
    chromosomes = ['chr'+str(i) for i in range(1,23)]

    merge_tdata=None #holds PCA on training data for 22 chromosomes
    merge_idata=None #holds PCA on independent data for 22 chromosomes

    max_workers = 10
    pm = ProcessManager()
    m = Manager()
    output=m.dict()
    for chrm in chromosomes:
        pm.add_process(do_PCA_subtask_with_kNN_impute, chrm,k,output)

    futs=pm.launch(max_workers=max_workers)
    print(wait(futs, return_when=futures.ALL_COMPLETED))
    print("Output:",len(output))
    for key,val in output.items():
        if merge_tdata is None:
            merge_tdata = val['training']
            print(key+" : training - After merge:" + str(merge_tdata.shape))
            merge_tdata = pd.concat([merge_tdata, val['t_labels']], axis=1)
            print(key+" : training - After adding labels:" + str(merge_tdata.shape))
        else:
            print(key+" : training - Merging with other chromosomes..")
            merge_tdata = pd.concat([merge_tdata, val['training']], axis=1)  # merge with other chromosomes
            print(key+" : training - After merge:" + str(merge_tdata.shape))
        if merge_idata is None:
            merge_idata = val['independent']
            print(key+" : Independent - After merge:" + str(merge_idata.shape))
            merge_idata = pd.concat([merge_idata, val['i_labels'],val['i_barcodes']], axis=1)
            print(key+" : Independent - After adding labels:" + str(merge_idata.shape))
        else:
            print(key+" : Independent - Merging with other chromosomes..")
            merge_idata = pd.concat([merge_idata, val['independent']], axis=1)  # merge with other chromosomes
            print(key+" : Independent - After merge:" + str(merge_idata.shape))

    print("Process complete")
    #print("Writing merged data to file...")

    return merge_tdata,merge_idata

def pipeline_model_selection(X_train,y_train,X_test,y_test):
    models_details={}
    #pipe_dt = Pipeline([('scl', StandardScaler()),('clf', DecisionTreeClassifier(random_state=42))])
    pipe_dt = Pipeline([('clf', DecisionTreeClassifier(random_state=42))])
    #pipe_rf = Pipeline([('scl', StandardScaler()),('clf', RandomForestClassifier(random_state=42))])
    pipe_rf = Pipeline([('clf', RandomForestClassifier(random_state=42))])
    #pipe_svm = Pipeline([('scl', StandardScaler()),('clf', svm.SVC(random_state=42))])
    pipe_svm = Pipeline([('clf', svm.SVC(random_state=42))])
    #pipe_linear_svm = Pipeline([('scl', StandardScaler()),('clf', svm.SVC(random_state=42))])
    pipe_linear_svm = Pipeline([('clf', svm.SVC(random_state=42))])

    # Set grid search params
    param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    param_range_fl = [1.0, 0.5, 0.1]

    grid_params_dt = [{'clf__criterion': ['gini', 'entropy'],
                       'clf__min_samples_leaf': param_range,
                       'clf__max_depth': param_range,
                       'clf__min_samples_split': param_range[1:]}]

    grid_params_rf = [{'clf__criterion': ['gini', 'entropy'],
                       'clf__min_samples_leaf': param_range,
                       'clf__max_depth': param_range,
                       'clf__min_samples_split': param_range[1:]}]

    #grid_params_svm = [{'clf__kernel': ['linear', 'rbf'],
    #                   'clf__C': param_range}]

    grid_params_svm = [{'clf__kernel': ['linear'],
                        'clf__C': [1,2]}]

    grid_params_Linear_svm = [{'clf__C': [1,2]}]

    # Construct grid searches
    jobs = -1

    gs_dt = GridSearchCV(estimator=pipe_dt,
                         param_grid=grid_params_dt,
                         scoring='accuracy',
                         cv=10,
                         n_jobs=jobs,
                         verbose=10)

    gs_rf = GridSearchCV(estimator=pipe_rf,
                         param_grid=grid_params_rf,
                         scoring='accuracy',
                         cv=10,
                         n_jobs=jobs,
                         verbose=10)

    gs_svm = GridSearchCV(estimator=pipe_svm,
                          param_grid=grid_params_svm,
                          scoring='accuracy',
                          cv=10,
                          n_jobs=jobs,
                          verbose=10)

    gs_linear_svm = GridSearchCV(estimator=pipe_linear_svm,
                                 param_grid=grid_params_Linear_svm,
                                 scoring='accuracy',
                                 cv=10,
                                 n_jobs=jobs,
                                 verbose=10)

    # List of pipelines for ease of iteration
    #grids = [gs_dt, gs_rf, gs_svm, gs_linear_svm]
    grids = [gs_svm, gs_linear_svm]
    # grids=[gs_dt]
    # Dictionary of pipelines and classifier types for ease of reference
    #grid_dict = {0: 'Decision Trees', 1: 'Random Forest',2: 'Support Vector Machine', 3: 'Linear Support Vector Machine'}

    grid_dict={0: 'Support Vector Machine', 1: 'Linear Support Vector Machine' }
    # Fit the grid search objects
    print('Performing model optimizations...')
    best_acc = 0.0
    best_clf = 0
    best_gs = ''
    for idx, gs in enumerate(grids):
        print('\nEstimator: %s' % grid_dict[idx])
        # Fit grid search
        gs.fit(X_train, y_train)
        # Best params
        print('Best params: %s' % gs.best_params_)
        # Best training data accuracy
        print('Best training accuracy: %.3f' % gs.best_score_)
        # Predict on test data with best params
        y_pred = gs.predict(X_test)
        # Test data accuracy of model with best params
        print('Test set accuracy score for best params: %.3f ' % accuracy_score(y_test, y_pred))
        # Track best (highest test accuracy) model
        models_details[grid_dict[idx]]={'best_params':str(gs.best_params_),
                                        'best_training_accuracy':str(gs.best_score_),
                                        'test_accuracy':str(accuracy_score(y_test, y_pred))}

        with open('/data2/han_lab/sravani/PCA/pipeline_results_scaling.txt', 'a') as f:
            f.write('\n\n\nEstimator: ' + str(grid_dict[idx]) +
                    '\nBest params: ' + str(gs.best_params_) +
                    '\nBest training accuracy: ' + str(gs.best_score_) +
                    '\nTest set accuracy score for best params: ' + str(accuracy_score(y_test, y_pred)) +
                    '\nModel filename: /data2/han_lab/sravani/PCA/' + '_'.join(
                grid_dict[idx].split()) + '_model.pkl')
        dump_file = '/data2/han_lab/sravani/PCA/' + '_'.join(grid_dict[idx].split()) + '_model.pkl'
        joblib.dump(best_gs, dump_file, compress=1)
        if accuracy_score(y_test, y_pred) > best_acc:
            best_acc = accuracy_score(y_test, y_pred)
            best_gs = gs
            best_clf = idx
    print('\nClassifier with best test set accuracy: %s' % grid_dict[best_clf])

    # Save best grid search pipeline to file
    dump_file = '/data2/han_lab/sravani/PCA/best_gs_pipeline.pkl'
    joblib.dump(best_gs, dump_file, compress=1)
    print('\nSaved %s grid search pipeline to file: %s' % (grid_dict[best_clf], dump_file))

    print("\nClassification Report by best model:")
    y_pred = best_gs.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

    with open('/data2/han_lab/sravani/PCA/best_pipeline_results_scalling.txt','w') as f:
        f.write('\n\n\nEstimator: ' + str(grid_dict[best_clf]) +
                '\nBest params: ' + str(best_gs.best_params_) +
                '\nBest training accuracy: ' + str(best_gs.best_score_) +
                '\nTest set accuracy score for best params: ' + str(accuracy_score(y_test, y_pred)) +
                '\nModel filename: /data2/han_lab/sravani/PCA' + '_'.join(
            grid_dict[best_clf].split()) + '_model.pkl'+
                '\nClassification Result: \n'+ str(report))

    best_params=str(best_gs.best_params_)
    model_name=grid_dict[best_clf]
    return best_gs,best_params,model_name,models_details

def process(tests,k,file,test_name,predicted_labels_file):
    # tcga_pca,idata_pca=get_PCA_data()


    if test=='mean_impute':
        tcga_pca, idata_pca = create_PCA_data_v3()
    if test=='kNN_impute':
        tcga_pca, idata_pca = create_PCA_data_v4(k)


    X = tcga_pca.drop(columns=['type'])
    y = tcga_pca['type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model=None
    if flag_pipeline_to_seclect_model:
        model, best_params, model_name,models_details = pipeline_model_selection(X_train, y_train, X_test, y_test)
        print("model:",model)
        print("best_params:",best_params)
        print("model_name:",model_name)
        print("model_details:",models_details)

        for algo,details in models_details.items():
            print("\n",algo)
            print(details)

        print("\n Best parameters:")
        print(best_params)
        print(model_name)

    else:

        from sklearn import svm
        model = svm.SVC(C=1, kernel='linear', random_state=42)

    model.fit(X_train, y_train)

    #fitting on test
    y_pred=model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

    accuracy0 = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy0)

    idata_X=idata_pca.drop(columns=['type','barcode'])
    idata_y=idata_pca['type']



    idata_y_pred = model.predict(idata_X)
    idata_report = classification_report(idata_y, idata_y_pred)
    print(idata_report)

    accuracy = accuracy_score(idata_y, idata_y_pred)
    print("Accuracy:", accuracy)

    model_name = 'Support Vector Machine'
    best_params = str({'clf__C': 1, 'clf__kernel': 'linear'})

    temp=pd.DataFrame()
    temp['actual']=idata_y
    temp['predicted']=idata_y_pred
    temp['barcode']=idata_pca['barcode']
    temp.to_csv(predicted_labels_file,sep=',')

    with open(file, 'a') as f:
        f.write("\n\n==========================================================================================")
        f.write("\n"+test_name)
        f.write("\nTraining Dataset\nBest model:" + model_name + '\nModel Parameters:' + best_params)
        f.write("\nClassification report on test data:\n")
        f.write(str(report))
        f.write('\n\nAccuracy:' + str(accuracy0))

        f.write("\n\n\nIndependent Dataset\nBest model:" + model_name + '\nModel Parameters:' + best_params)
        f.write("\nClassification report on independent data:\n")
        f.write(str(idata_report))
        f.write('\n\nAccuracy:' + str(accuracy))

def process_for_running_pipeline(tests,k,file,test_name):
    # tcga_pca,idata_pca=get_PCA_data()


    if test=='mean_impute':
        tcga_pca, idata_pca = create_PCA_data_v3()
    if test=='kNN_impute':
        tcga_pca, idata_pca = create_PCA_data_v4(k)


    X = tcga_pca.drop(columns=['type'])
    y = tcga_pca['type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model=None
    print("Running pipeline to select model...")
    model, best_params, model_name,models_details = pipeline_model_selection(X_train, y_train, X_test, y_test)

    for algo,details in models_details.items():
        print("\n",algo)
        print(details)

    print("\n Best parameters:")
    print(best_params)
    print(model_name)



t1=time.time()
#tests=['mean_impute','kNN_impute']
tests=['kNN_impute']
k=[1]
results_file='/data2/han_lab/sravani/PCA/PCA_k_1_results.txt'
predicted_labels_file='/data2/han_lab/sravani/PCA/labels_predicted_k_1.csv'

flag_standard_scaling=False
flag_pipeline_to_seclect_model=False #true to run pipeline, false works on best model
for test in tests:

    if test == 'mean_impute':
        test_name = 'Missing Value imputation in independent dataset with mean values'
        process(tests, k, results_file,test_name,predicted_labels_file)
    if test == 'kNN_impute':
        for val in k:
            test_name = 'Missing Value imputation in independent dataset with kNN where k='+str(val)
            process(tests, val, results_file,test_name,predicted_labels_file)

print("time take:",time.time()-t1)


import os
cmd = 'scp '+predicted_labels_file+' gannavar@bobby.cs.unlv.edu://home/gannavar/UNLV_Thesis/'
os.system(cmd)
cmd = 'scp '+results_file+' gannavar@bobby.cs.unlv.edu://home/gannavar/UNLV_Thesis/'
os.system(cmd)
'''
import subprocess
p = subprocess.Popen(["scp", "my_file.txt", "username@server:path"])
sts = os.waitpid(p.pid, 0)
'''


'''

'''