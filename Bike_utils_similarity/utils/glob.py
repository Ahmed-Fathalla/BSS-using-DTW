import numpy as np
import pandas as pd
from glob import glob
import os


# read dfs
for ds in sorted(glob('data/*2020.csv')): 
    df = pd.read_csv(ds)
    dd.to_csv(file_name)
    
# rename files
for ds in sorted(glob('data/*.csv')): 
    new_name = 'data/'+ds.split('/')[1].split('_')[0]+'.csv'
    os.rename(ds,new_name)    

# Other tasks
number_of_splits = 2
from glob import glob
for ds in sorted(glob('data/*2020.csv')): 
    # for i in range(1,31):
    ds = 'data/%d_2020.csv'%i
    # print(ds) 
    df = pd.read_csv(ds)
    len_ = df.shape[0]
    bulk_size = int(len_/number_of_splits)
    for i in range(number_of_splits):
        dd = df[i*bulk_size:(i+1)*bulk_size]
        
        file_name = 'data/exp/%s-%d.csv'%(ds[5:-4],i)
        # print(file_name, '\ndd = ' , dd )
        dd.to_csv(file_name)
        # sh.append(dd.shape)
        print(file_name, dd.shape)
        
        
        
        
        