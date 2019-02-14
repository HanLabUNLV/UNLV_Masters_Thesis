import hypertools as hyp
import numpy as np
import scipy
import pandas as pd
from scipy.linalg import toeplitz
from copy import copy

data=pd.read_csv('/home/sravani/UNLV_Thesis/PCA/IData_combined_PCA_data.csv')

labels=list(data['TCGA Label '])
len(labels)
labels[0:5]

unique=list(data['TCGA Label '].unique())
len(unique)
# plot
geo = hyp.plot(data, '.',hue=labels,size=[12,8],animate='spin',legend=unique,zoom=3.5,save_path='IDATA_PCA_zoom_animation.mp4')
geo = hyp.plot(data, '.',hue=labels,size=[12,8],animate='spin',legend=unique,save_path='IDATA_PCA_animation.mp4') # plots dots
#,save_path='animation.mp4'