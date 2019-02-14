
from multiprocessing import Manager
from os import walk
import os
import pandas as pd
from os import walk
import os
import multiprocessing as mp

# now just give your processes access to ns, i.e. most simply
# p = Process(target=worker, args=(ns, work_unit))


f=[]
IData_path="/data2/han_lab/sravani/Data/IndependentData/"
for (dirpath, dirnames, filenames) in walk(IData_path):
    for name in filenames:
        if "_probe_data.csv" in name:
            f.extend([name])


print(f)

probe_names=pd.read_csv('/data2/han_lab/sravani/missing_counts.csv',sep='\t')
chromosome_list=['chr'+str(i) for i in range(1,23)]

#split probe_names into multiple dataframes per chromosome
chr_collection={}
for chr in chromosome_list:
    chr_collection[chr]=probe_names[probe_names['chr']==chr]

def map_chromosome(ns,name,path,chr,probes,output):
    print name,chr
    probes = probes.drop(columns=[u'counts'])
    x = ns.df.loc[ns.df[u'ID_REF'].isin(probes[u'Composite'])]
    x = x.set_index(u'ID_REF')
    if probes.shape[0]==x.shape[0]:
        print name+'_'+chr+'_actual: ', probes.shape
        print name+'_'+chr +'_curr: ', x.shape
    else:
        print name+'_'+chr+' probe count mismatch'
    probes = probes.set_index(u'Composite')
    y = pd.concat([x, probes], axis=1,sort=True)
    y = y.reset_index()
    y = y.rename(columns={'index': 'composite'})
    y.to_csv(path+name+'_'+chr,sep=',',index=False,)
    output.put(path+name+'_'+chr)

for  file in f:
    print("\n============================================\n")
    print(file)
    name=file.split('_')[0]
    path='/data2/han_lab/sravani/Data/IndependentData/per_chromosome/'
    df = pd.read_csv(IData_path+file, sep=',')
    print df.shape
    missing_probes = []
    #method to map probes and chromosomes

    if df[u'"ID_REF"'].str.contains('!series_matrix_table_end').any():
        temp = df[u'"ID_REF"'].str.find('!series_matrix_table_end')
        idx = temp[temp == 0.0].index[0]
        df = df.iloc[0:idx, :]
        print df.shape
        df['ID_REF'] = df[u'"ID_REF"'].str.strip('"')
        df = df.drop(columns=['"ID_REF"'])
        df.columns = df.columns.str.strip('"')
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df = df[cols]
        # df = df.set_index(['ID_REF'])
        # multi processing
        mgr = Manager()
        ns = mgr.Namespace()
        ns.df = df
        output = mp.Queue()
        output=mp.Queue()
        processes = [mp.Process(target=map_chromosome, args=(ns,name,path,'chr'+str(x),chr_collection['chr' + str(x)], output)) for x in
                     range(1, 23)]
        # Run processes
        for p in processes:
            p.start()

        # Exit the completed processes
        for p in processes:
            p.join()

        # Get process results from the output queue
        results = [output.get() for p in processes]

        print(results)
    else:
        continue

chromosome_list=['chr'+str(i) for i in range(1,23)]
chr_files={}
IData_path="/data2/han_lab/sravani/Data/IndependentData/per_chromosome"
for (dirpath, dirnames, filenames) in walk(IData_path):
    for chr in chromosome_list:
        chr_files[chr]=[]
        for name in filenames:
            if name.split('_')[1]==chr:
                chr_files[chr].extend([name])

import pandas as pd
for chr in chromosome_list:
    merge=None
    print chr
    for f in chr_files[chr]:
        df=pd.read_csv(IData_path+"/"+f,sep=',')
        print(df.shape)
        df=df.set_index([u'composite'])
        if merge is None:
            merge=df
        else:
            merge=pd.concat([merge,df],axis=1,sort=False)
            print 'merge: ',merge.shape
    merge.to_csv(IData_path+'/'+chr+'.csv',sep=',')
    print IData_path+'/'+chr+'.csv'

'''
probes=chr_collection['chr1']
file="/data2/han_lab/sravani/Data/IndependentData/GSE38268_probe_data.csv"
df=pd.read_csv(file,sep=',')
#df['chromosome']='missing'
temp = df[u'"ID_REF"'].str.find('!series_matrix_table_end')
idx = temp[temp == 0.0].index[0]
df = df.iloc[0:idx, :]
print df.shape
df['ID_REF'] = df[u'"ID_REF"'].str.strip('"')
df = df.drop(columns=['"ID_REF"'])
df.columns = df.columns.str.strip('"')

cols=df.columns.tolist()
cols=cols[-1:]+cols[:-1]
df=df[cols]


probes=probes.drop(columns=[u'counts'])

x=df.loc[df[u'ID_REF'].isin(probes[u'Composite'])]
x=x.set_index(u'ID_REF')
probes=probes.set_index(u'Composite')
y=pd.concat([x,probes],axis=1)
y=y.reset_index()
y=y.rename(columns={'index':'composite'}
'''