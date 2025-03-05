from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings, sys
warnings.filterwarnings('ignore')
def pdng(str_, i, padding_char = ' '):
    return str_+ padding_char*(i-len(str_))
    
    
def handle(x):
    a = x.split('/')[0].replace('yr','').replace('y','').replace('m','').replace(' ','').split(',')
    a = [int(i) for i in a]
    return a[0]*12+a[1] if len(a)>1 else a[0]*12
    
def upper_case():
    from string import ascii_uppercase    
    l = []
    for i in ['',*ascii_uppercase[:3]]:
        for j in ascii_uppercase:
            l.append(i+j)

def get_data(lst, target):
    ind_=None
    for count,l in enumerate(lst):
        # print(l, count)
        if l == target:
            ind_ = count
            break
    return lst[:ind_+1]


def get_categorical_features(x,thresholds_ratio=0.05, verboseunique=0):
    df_len = x.shape[0]
    categorical_features = []
    for i in x.columns:
        unique = x[i].nunique()
        if unique/df_len < thresholds_ratio:
            categorical_features.append([i,unique])
            if verboseunique: print('%-20s'%i,x[i].nunique() )
    return np.array(categorical_features) 

def hello_df(df, thresholds_ratio=1.1, save_plt=False, show_plt=True, verbose=1):
    '''
    parameters:
        df: DataFrame to display its data
    just like .describe(), but 
        1. display the results for categorical variables only. 
        2. return the cols that have 'object' dtypes
    '''
    max_len = 0
    for i in df.columns.values:
        if len(i)> max_len:
            max_len=len(i)
    
    df = df.copy()
    print( df.shape )
    categorical_features = get_categorical_features(df,thresholds_ratio=thresholds_ratio)
    # print(categorical_features)
    if verbose:
        print( '='*(max_len+35) ,'\n%s\t'%pdng('Col_Name',max_len),'      DataType',' NAN     %')
        print('='*(max_len+35))
    
    for i,j in zip(df.dtypes.index,df.dtypes):
        s = ''
        # if j == 'object':s = '%-15s----- '%i
        if i in categorical_features[:,0]:
            s = '%s  %-5d--- '%(pdng(i,max_len+1),df[i].nunique())
        else:
            s='%s           '%pdng(i,max_len+1) 
        
        s += '%-10s'%j
        s += '%-5d(%-5s%%)'%(df[i].isnull().sum(), str(df[i].isnull().sum() * 100/ len(df))[:5])
        if verbose:print(s)
        
    continuous_features = [i for i in df.columns if i not in categorical_features]
    
    if verbose:print('='*30,'\n'*2)
    
    null_percent = get_nulls(df, save_plt=save_plt, show_plt=show_plt, verbose=verbose)

    # from IPython.display import display, HTML
    # display(HTML(df[df[categorical_features[:,0]]].describe().to_html()))
    
    return categorical_features, continuous_features, null_percent # list(df.select_dtypes(include=['category','object']))





# def max_len(lst):
    # len_ = 0
    # for i in lst:
        # if len(str(i))>len_:
            # len_= len(str(i))
    # return len_
            
# def print_dic(dic):
    # alignment_len_ = max_len(dic.keys())
    # for k,v in dic.items():
        # print('\t%s: %s'%(pdng(k, alignment_len_),str(v)))