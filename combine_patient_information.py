
import pandas as pd
import numpy as np

from os import walk


def rn(df, suffix = '_dp_'):
    appendents = (suffix + df.groupby(level=0).cumcount().astype(str).replace('0','')).replace(suffix, '')
    return df.set_index(df.index + appendents)

f = []
def independent():
d1=None
merge=None
idata_path = '/data2/han_lab/sravani/Data/IndependentData/'
for (dirpath, dirnames, filenames) in walk(idata_path):
    #print(filenames)
    for file in filenames:

        if file.endswith('_info.csv'):
            #file=file.encode('utf-8')
            f=idata_path+file
            d1=pd.read_csv(f,sep=',',header=None)
            print("file:"+str(file)+"\n",d1.iloc[:,0:2])
            d1 = d1.set_index(0)
            d1=rn(d1)

            #print(d1)
            #d1.index = d1.index.where(~d1.index.duplicated(), d1.index + '_dp')
            #print(d.iloc[0:5, 0:5])
            if merge is None:
                merge = d1
                print("merge:",merge.shape)
            else:
                print("d1:" + str(d1.shape) + " merge:" + str(merge.shape))
                merge = pd.concat([merge, d1], axis=1,sort=False)
                print("merge:", merge.shape)

    #print("\njust_done",merge)

    #print(merge1.iloc[0:5,0:5])
    merge1=merge.T
    #merge1=merge1.reset_index()
    #print(merge1.iloc[0:5,0:5])
    #merge1 = merge1.set_index('!Sample_geo_accession')
    #merge1.columns = merge1.iloc[0]
    #merge1=merge1.reindex(merge1.index.drop(0))
    #print(merge1.iloc[0:5,0:5])
    merge2=merge1.set_index('!Sample_geo_accession')
    #removing duplicate values in index
    #df3 = merge2[~merge2.index.duplicated(keep=False)]

    idata_labels = pd.read_csv('/data2/han_lab/sravani/Data/IndependentData/independent_data_labels.csv', sep=',')
    idata_labels = idata_labels.set_index('Sample ID')

    #extracting only required probes from labels
    x = idata_labels[idata_labels.index.isin(merge2.index)]

    #merge with labels
    data = pd.concat([merge2, x], axis=1, sort=False)
    data = data.rename(columns={"TCGA Label ": "type"})
    data=data.reset_index()
    data.to_csv('/data2/han_lab/sravani/Data/independent_patient_details.csv', sep=',', index=False)
    return data

d=independent()
d.to_csv('/data2/han_lab/sravani/Data/IndependentData/independent_patient_info.csv',sep=',',index=False)

def training():
    d=pd.read_csv('/data2/han_lab/TCGA/methylation/gdc_sample_sheet.2018-09-10.tsv',sep='\t')
    temp=pd.DataFrame()
    new=d['File Name'].str.split('.',n=6,expand=True)
    temp['barcode']=new[5]
    temp['sample_type']=d['Sample Type']
    temp=temp.set_index('barcode')
    labels = pd.read_csv('/data2/han_lab/sravani/Data/patientToCancerLabels_final.csv', sep=',')
    labels = labels.set_index('Barcode1')
    merge=pd.concat([temp,labels],axis=1,sort=False)
    merge=merge.reset_index()
    return merge

d1=training()
d1.to_csv('/data2/han_lab/sravani/Data/training_patient_info.csv',sep=',')



