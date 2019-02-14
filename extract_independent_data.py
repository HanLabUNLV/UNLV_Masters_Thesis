
from os import walk
import pandas as pd

path="/data2/han_lab/GEO.cancer/methylation/"

f = []
for (dirpath, dirnames, filenames) in walk(path):
    for name in filenames:
        if "series_matrix.txt" in name:
            f.extend([name])


print(f)

files=[path+x for x in f]
x=None
IData_path="/data2/han_lab/sravani/Data/IndependentData/"
for  file in files:
    print("\n============================================\n")
    print(file)
    code=file.split('/')[-1].split('_')[0]
    probe_file=IData_path+code+"_probe_data.csv"
    info_file=IData_path+code+"_info.csv"
    with open(file, 'r') as f:
        full = f.read()
    full1 = full.split('\n')
    print(len(full1))
    # get info details
    sample_details = []
    num=-1
    end=-1
    for i in range(len(full1)):
        if full1[i].startswith("!Sample_"):
            sample_details.extend([full1[i]])
        elif full1[i].startswith("!series_matrix_table_begin"):
            num=i
        elif full1[i].startswith("!series_matrix_table_end"):
            end=i
            break
    '''
    for i in range(len(sample_details)):
        sample_details[i] = sample_details[i].split('\t')
        sample_details[i]=[s.strip('\'"') for s in sample_details[i]]
    x=sample_details

    df1 = pd.DataFrame(sample_details)
    df1.to_csv(info_file, sep=',',header=None, index=False)
    print("Info shape:",df1.shape)
    print("Info file: "+info_file)
    '''
    #get probe data
    probe=full1[num+1:end]
    for i in range(len(probe)):
        probe[i] = probe[i].split('\t')

    df = pd.DataFrame(probe)
    df.to_csv(probe_file, sep=',',header=None, index=False)
    print("probe shape:",df.shape)
    print("Probe file: "+probe_file)
