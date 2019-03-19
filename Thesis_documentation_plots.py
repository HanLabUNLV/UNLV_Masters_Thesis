
import pandas as pd
import matplotlib.pyplot as plt
import os

training_labels = pd.read_csv('/data2/han_lab/sravani/Data/patientToCancerLabels_final.csv', sep=',')
training_labels = training_labels.set_index('Barcode1')

d = pd.read_csv('/data2/han_lab/sravani/Data/without_NA/chr22_withoutNA.csv')
d = d.set_index('Composite')
dT = d.T  # transpose it

# merge dfT and labels
tcga_original = pd.concat([dT, training_labels], axis=1)


tcga_filename='/data2/han_lab/sravani/version2/datasets/TCGA_SelectKBest_k50.csv'
idata_filename='/data2/han_lab/sravani/version2/datasets/Idata_kNN_impute_k20_neigh_5.csv'

tcga=pd.read_csv(tcga_filename,sep=',')

# remove samples with type count <100
dist_original=tcga_original.groupby(['type']).size()
dist_original=dist_original.sort_values(ascending=False)
print(dist_original)

dist_filter = tcga.groupby(['type']).size().sort_values(ascending=False)
dist_filter=dist_filter.sort_values(ascending=False)
print(dist_filter)

# remove samples with type count <100
dist_original=tcga_original.groupby(['type']).size()
dist_original=dist_original.sort_values(ascending=False)
print(dist_original)

dist_filter = tcga.groupby(['type']).size()
dist_filter=dist_filter.sort_values(ascending=False)
print(dist_filter)


#independent data
idata_labels = pd.read_csv('/data2/han_lab/sravani/Data/IndependentData/independent_data_labels1.csv', sep=',')
idata_labels = idata_labels.set_index('Sample ID')
df=pd.read_csv('/data2/han_lab/sravani/Data/IndependentData/per_chromosome/chr22.csv')

df = df.set_index([u'composite'])
test = df.T
x=idata_labels[idata_labels.index.isin(test.index)]
data = pd.concat([test, x], axis=1,sort=False)
print(data.shape)

idist_original=data.groupby(['TCGA Label ']).size()
idist_original=idist_original.sort_values(ascending=False)

idata=pd.read_csv(idata_filename,sep=',')
import numpy as np
req_labels=np.unique(tcga['type'])

data1 = data[data['TCGA Label '].isin(req_labels)]
temp=data1.groupby(['TCGA Label ']).size()
temp=temp.sort_values(ascending=False)


IData_path = '/data2/han_lab/sravani/Data/IndependentData/per_chromosome/'
file = IData_path + 'chr22' + '.csv'
df = pd.read_csv(file, sep=',')






ax = dist_original.plot(kind='bar', title ="Distribution of class labels", figsize=(15, 10),  fontsize=12)
ax.set_xlabel("Class labels", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
plt.show()
fig = ax.get_figure()
fig.savefig('/data2/han_lab/sravani/version2/pipeline/TCGA_original_distributon.png')
cmd = 'scp '+'/data2/han_lab/sravani/version2/pipeline/TCGA_original_distributon.png'+' gannavar@bobby.cs.unlv.edu://home/gannavar/UNLV_Thesis/'
os.system(cmd)


ax = dist_filter.plot(kind='bar', title ="Distribution of class labels", figsize=(15, 10), fontsize=12)
ax.set_xlabel("Class labels", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
plt.show()
fig = ax.get_figure()
fig.savefig('/data2/han_lab/sravani/version2/pipeline/TCGA_filter_distributon.png')
cmd = 'scp '+'/data2/han_lab/sravani/version2/pipeline/TCGA_filter_distributon.png'+' gannavar@bobby.cs.unlv.edu://home/gannavar/UNLV_Thesis/'
os.system(cmd)

filter = temp[temp[0] > 100]
train_filter = data[data['type'].isin(filter['type'])]