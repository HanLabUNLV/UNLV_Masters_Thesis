import json
import pandas as pd
import pickle
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from multiprocessing import Manager
import multiprocessing as mp
from sklearn.pipeline import Pipeline
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

def get_train_data(chrm,training_labels):
    # load the labels
    #get without NA data

    d = pd.read_csv('/data2/han_lab/sravani/Data/without_NA/' + chrm + '_withoutNA.csv')
    d = d.set_index('Composite')
    dT = d.T  # transpose it
    print "TCGA withoutNA Shape:" + str(dT.shape)

    # drop probes with mean <0.2
    dT = dT[dT.columns[dT.mean(axis=0) > 0.2]]
    print "TCGA after removing row means <0.2 Shape:" + str(dT.shape)

    # merge dfT and labels
    data = pd.concat([dT, training_labels], axis=1)
    print "Training data shape:"+ str(data.shape)

    # drop few uneccesary columns
    X = data.reset_index().drop(columns=['Barcode', 'Label', 'Barcode1', 'TSS', 'Center', 'Plate', 'type'])
    y = data.reset_index()['type']
    probes=[i for i in X.columns]
    return X,y,probes

def get_independent_data(chrm,idata_labels,probes):
    IData_path = '/data2/han_lab/sravani/Data/IndependentData/per_chromosome/'
    file = IData_path + chrm + '.csv'
    df = pd.read_csv(file, sep=',')
    df1 = df.drop(columns=['chr.' + str(i) for i in range(1, 21)])
    df1 = df1.drop(columns=['chr'])
    df2 = df1[df1[u'composite'].isin(probes)]
    df2 = df2.set_index([u'composite'])
    test = df2.T
    x=idata_labels[idata_labels.index.isin(test.index)]
    data = pd.concat([test, x], axis=1)
    print "Independent data shape:"+ str(data.shape)
    '''
    #display rows with Null values
    df1 = data[data.isnull().any(axis=1)]
    #find if any lables are missing
    temp=data[data.isnull().any(axis=1)]
    null_index=temp[temp['TCGA Label '].isnull()].index.tolist()
    idata_labels.ix[null_index]  
    '''
    X = data.reset_index().drop(columns=['TCGA Label ', 'Tissue Type', 'index'])
    y = data.reset_index()['TCGA Label ']
    idx=[i for i in probes]
    X=X[idx]
    return X,y

def fill_missing_mean(df):
    if df.isnull().values.any():

        #fill missing values with column means( probe mean)
        return df.fillna(df.mean())

def create_PCA_data():
    #if __name__==__main__():
    chromosomes = ['chr'+str(i) for i in range(1,23)]

    merge_tdata=None #holds PCA on training data for 22 chromosomes
    merge_idata=None #holds PCA on independent data for 22 chromosomes
    for chrm in chromosomes:
        print("===========================================================================")
        print("chromosome:",chrm)
        print("loading training labels...")
        training_labels= pd.read_csv('/data2/han_lab/sravani/Data/patientToCancerLabels_final.csv', sep=',')
        training_labels = training_labels.set_index('Barcode1')

        #probes : those probes considered for PCA
        print("Getting training data...")
        train_X,train_y,probes = get_train_data(chrm,training_labels)

        print("")

        print("Loading independent data labels..")
        idata_labels = pd.read_csv('/data2/han_lab/sravani/Data/IndependentData/independent_data_labels.csv', sep=',')
        idata_labels = idata_labels.set_index('Sample ID')
        print("Getting independent data..")
        idata_X,idata_y = get_independent_data(chrm,idata_labels,probes)

        if idata_X.shape[1]==train_X.shape[1]:
            if idata_X.isnull().values.any():
                print("Independent Data has missing values..")
                print("filling missing values with mean..")
                idata_X_new=fill_missing_mean(idata_X)

                print("")
                print("Fitting training data to model..")

                model = PCA(n_components=100)
                model.fit(train_X)

                print("")
                print("Transforming TCGA data according to model..")
                train_X_trans=model.transform(train_X)
                train_X_trans=pd.DataFrame(train_X_trans)
                train_X_trans=train_X_trans.add_prefix(str(chrm+'_PC_'))


                if merge_tdata is None:
                    merge_tdata=train_X_trans
                    print("After merge:"+str(merge_tdata.shape))
                    merge_tdata = pd.concat([merge_tdata, train_y], axis=1)
                    print("After adding labels:" + str(merge_tdata.shape))
                else:
                    print("Merging with other chromosomes..")
                    merge_tdata=pd.concat([merge_tdata,train_X_trans],axis=1) #merge with other chromosomes
                    print("After merge:"+str(merge_tdata.shape))

                print("")
                print("Transforming independent data according to model..")
                idata_X_trans=model.transform(idata_X_new)
                idata_X_trans=pd.DataFrame(idata_X_trans) #convert numpy array to dataframe
                idata_X_trans=idata_X_trans.add_prefix(str(chrm+'_')) # give column names
                if merge_idata is None:
                    merge_idata=idata_X_trans
                    print("After merge:"+str(merge_idata.shape))
                    merge_idata = pd.concat([merge_idata, idata_y], axis=1)
                    print("After adding labels:" + str(merge_idata.shape))
                else:
                    print("Merging with other chromosomes..")
                    merge_idata=pd.concat([merge_idata,idata_X_trans],axis=1) #merge with other chromosomes
                    print("After merge:"+str(merge_idata.shape))


        else:
            print("Number of probes mismatch in Independent Data and train data.")

    print("Process complete")
    print("Writing merged data to file...")
    merge_idata.to_csv('/data2/han_lab/sravani/PCA/IData_combined_PCA_data.csv',index=False)
    merge_tdata.to_csv('/data2/han_lab/sravani/PCA/TCGA_combined_PCA_data.csv',index=False)

    return merge_tdata,merge_idata


def get_PCA_data():
    idata_pca=pd.read_csv('/data2/han_lab/sravani/PCA/IData_combined_PCA_data.csv', sep=',')
    tcga_pca=pd.read_csv('/data2/han_lab/sravani/PCA/TCGA_combined_PCA_data.csv', sep=',')
    return tcga_pca,idata_pca

def pipeline_model_selection(X_train,y_train,X_test,y_test):
    pipe_dt = Pipeline([('scl', StandardScaler()),
                        ('clf', DecisionTreeClassifier(random_state=42))])

    pipe_rf = Pipeline([('scl', StandardScaler()),
                        ('clf', RandomForestClassifier(random_state=42))])

    pipe_svm = Pipeline([('scl', StandardScaler()),
                         ('clf', svm.SVC(random_state=42))])

    pipe_linear_svm = Pipeline([('scl', StandardScaler()),
                                ('clf', svm.SVC(random_state=42))])

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

    grid_params_svm = [{'clf__kernel': ['linear', 'rbf'],
                        'clf__C': param_range}]

    grid_params_Linear_svm = [{'clf__C': param_range}]

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
    grids = [gs_dt, gs_rf, gs_svm, gs_linear_svm]
    # grids=[gs_dt]
    # Dictionary of pipelines and classifier types for ease of reference
    grid_dict = {0: 'Decision Trees', 1: 'Random Forest',
                 2: 'Support Vector Machine', 3: 'Linear Support Vector Machine'}
    grid_dict={0: 'Decision Trees', 1: 'Random Forest',
                 2: 'Support Vector Machine', 3: 'Linear Support Vector Machine'}
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
        with open('/data2/han_lab/sravani/PCA/pipeline_results.txt', 'a') as f:
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

    with open('/data2/han_lab/sravani/PCA/best_pipeline_results.txt','w') as f:
        f.write('\n\n\nEstimator: ' + str(grid_dict[best_clf]) +
                '\nBest params: ' + str(best_gs.best_params_) +
                '\nBest training accuracy: ' + str(best_gs.best_score_) +
                '\nTest set accuracy score for best params: ' + str(accuracy_score(y_test, y_pred)) +
                '\nModel filename: /data2/han_lab/sravani/PCA' + '_'.join(
            grid_dict[best_clf].split()) + '_model.pkl'+
                '\nClassification Result: \n'+ str(report))

    best_params=str(best_gs.best_params_)
    model_name=grid_dict[best_clf]
    return best_gs,best_params,model_name


#tcga_pca,idata_pca=get_PCA_data()
tcga_pca,idata_pca=create_PCA_data()

#removing the labels whose count is less than 100 before creating the model
temp=tcga_pca.groupby(['type']).size().reset_index()
filter=temp[temp[0]>100]
filter_data=tcga_pca[tcga_pca['type'].isin(filter['type'])]
print(filter_data.groupby(by=['type']).size())

# Split into X and Y
X =filter_data.drop(columns=['type'])
y = filter_data['type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_labels=pd.DataFrame()
test_labels=pd.DataFrame()
train_labels['type'],train_labels['count']=np.unique(y_train,return_counts=True)
test_labels['type'],test_labels['count']=np.unique(y_test,return_counts=True)
print("\nTrain:")
print(train_labels)
print("\nTest:")
print(test_labels)


model,best_params,model_name=pipeline_model_selection(X_train,y_train,X_test,y_test)

'''
#load model from file
import pickle
filename='/data2/han_lab/sravani/PCA/best_gs_pipeline.pkl'
#model = pickle.load(open('/data2/han_lab/sravani/PCA/best_gs_pipeline.pkl', 'rb'))
with open(filename, 'rb') as file:
    pickle_model = pickle.load(file)


from sklearn import svm
model1=svm.SVC(C=1,kernel='linear',random_state=42)
model1.fit(X_train,y_train)
idata_y_pred = model1.predict(idata_X)
idata_report = classification_report(idata_y, idata_y_pred)
print(idata_report)

accuracy=accuracy_score(idata_y, idata_y_pred)
print("Accuracy:",accuracy)
model_name='Support Vector Machine'   
best_params= str({'clf__C': 1, 'clf__kernel': 'linear'})
with open('/data2/han_lab/sravani/PCA/independent_data_classification_report.txt','w') as f:
    f.write("\nIndependent Dataset\nBest model:"+model_name+'\nModel Parameters:'+best_params)
    f.write("\nClassification report on independent data:\n")
    f.write(str(idata_report))
    f.write('\n\nAccuracy:'+str(accuracy))
'''

idata_X=idata_pca.drop(columns=['TCGA Label '])
idata_y=idata_pca['TCGA Label ']
idata_y_pred = model.predict(idata_X)
idata_report = classification_report(idata_y, idata_y_pred)
print(idata_report)

accuracy=accuracy_score(idata_y, idata_y_pred)

with open('/data2/han_lab/sravani/PCA/independent_data_classification_report.txt','w') as f:
    f.write("\nIndependent Dataset\nBest model:"+model_name+'\nModel Parameters:'+best_params)
    f.write("\nClassification report on independent data:\n")
    f.write(str(idata_report))
    f.write('\n\nAccuracy:'+str(accuracy))

