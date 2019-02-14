
'''
This code fills in the missing values
'''
#get complete list of labels after replacing actual values with toDecide
a=pd.read_csv('/home/sravani/UNLV_Thesis/patientToCancerLabels.csv',sep='\t')
b=pd.read_csv('/home/sravani/UNLV_Thesis/missing_mapped.csv',sep=',')

temp2=b[['submittedAliquot ID','mappedProject']]
temp2['type']=temp2['mappedProject'].str.split('-').str[1]
temp2=temp2.drop(columns=['mappedProject'])
temp2=temp2.set_index('submittedAliquot ID')
for i in range(len(a)):
    if a.loc[i,'type']=='toDecide':
        item=a.loc[i,'Barcode1']
        if item in temp2.index:
            a.loc[i,'type']=temp2.loc[item,'type']
            print (item)
temp1=a[a['type']=='toDecide']

a.to_csv('/home/sravani/UNLV_Thesis/patientToCancerLabels_final.csv',sep='\t')