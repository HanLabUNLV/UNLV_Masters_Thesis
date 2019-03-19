import pandas as pd
probe_positions=pd.read_csv('/data2/han_lab/TCGA/methylation/probe.TCGA.illumina.txt',sep='\t')
tcga_filename=pd.read_csv('/data2/han_lab/sravani/version2/datasets/TCGA_SelectKBest_k50.csv',sep=',')


colnames=[i for i in tcga_filename.columns]
colnames.remove('Barcode1')
colnames.remove('type')
temp=pd.DataFrame()
temp['req_probes']=colnames
temp=temp.set_index('req_probes')


temp2=probe_positions[probe_positions['ID'].isin(colnames)]
temp2=temp2.sort_values(by=['CHR'])
temp2.to_csv('/data2/han_lab/sravani/version2/datasets/required_probe_positions.csv',index=False)


import os
cmd = 'scp '+'/data2/han_lab/sravani/version2/datasets/required_probe_positions.csv'+' gannavar@bobby.cs.unlv.edu://home/gannavar/UNLV_Thesis/'
os.system(cmd)

probe_positions=pd.read_csv('/data2/han_lab/sravani/version2/datasets/required_probe_positions.csv',sep=',')

probe_scores=pd.read_csv(r'/data2/han_lab/sravani/version2/pipeline/probe_scores.txt',header=None, sep=' ')
req_probe_scores=probe_scores[probe_scores[0].isin(probe_positions['ID'])]
req_probe_scores=req_probe_scores.drop_duplicates(subset=0,keep='first')
req_probe_scores=req_probe_scores.set_index(0)
probe_positions=probe_positions.set_index('ID')
merge=pd.concat([req_probe_scores,probe_positions],sort=False,axis=1)
merge.shape
merge=merge.rename(columns={1:'score'})
merge.to_csv('/data2/han_lab/sravani/version2/datasets/required_probe_positions.csv',sep='\t')

import os
cmd = 'scp '+'/data2/han_lab/sravani/version2/datasets/required_probe_positions.csv'+' gannavar@bobby.cs.unlv.edu://home/gannavar/UNLV_Thesis/'
os.system(cmd)
