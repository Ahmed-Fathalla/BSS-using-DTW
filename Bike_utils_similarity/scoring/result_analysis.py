#
# D:\Saves\0 Esraa\7 Bike\00 Bike paper - project Ali\utils
# D:\Saves\0 Esraa\7 Bike\00 Bike paper - project Ali\Results Analysis .ipynb


import pandas as pd
import numpy as np
from ..ML.exp_conf import exp_dict

def get_df(df, key=None, value=None,response=None, cols=None):
    if key==None and value==None:
        df=df.loc[ df.response==response ]
        return df[cols]

    if cols==None:
        return df.loc[ (df.key==key) & (df.value==value) & (df.response==response) ]
    elif cols in ['tst_MAE', 'tst_RMSE', 'tst_MSE', 'tst_R2', 'tst_MAPE']:
        return df.loc[(df.key == key) & (df.value == value) & (df.response == response)][cols]
    elif cols=='all':
        cols=['tst_MAE','tst_RMSE','tst_MSE','tst_R2','tst_MAPE']
        return df.loc[ (df.key==key) & (df.value==value) & (df.response==response) ][cols]

def get_matrix(al, sp, k_,res_,met_):
    res_df = pd.DataFrame()
    for v in exp_dict[k_]:
        res_df['%s'%str(v)] = get_df(sp, k_, v, res_, met_).values
    res_df['full'] = get_df(al, None, None, res_, met_).values
    return res_df

def heat_map(al, sp, k_, res_, met_, title=''):
    r = get_matrix(al, sp, k_, res_, met_)
    w = r.values
    tmp = r.copy()
    tmp['model'] = ['model_%d_orig' % i for i in range(1, 1 + tmp.shape[0])]
    empty_row_df = pd.DataFrame(np.array(['' for _ in range(tmp.shape[1])]).reshape(1,-1), columns=tmp.columns)
    tmp = pd.concat([tmp, empty_row_df])
    tmp = tmp[['model', *r.columns.tolist()]]
    for m in range(w.shape[0]):
        record = w[m, :].tolist()
        matrix = np.array([record for _ in range(w.shape[1])])
        new_matrix = np.zeros(matrix.shape)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                new_matrix[i, j] = (matrix[i, i] - matrix[i, j]) * 100 / matrix[i, i]
        z_df = pd.DataFrame(new_matrix, columns=r.columns)
        z_df['model'] = 'model_%d' % (m + 1)
        z_df = z_df[['model', *r.columns.tolist()]]
        tmp = pd.concat([tmp, z_df])
        tmp = pd.concat([tmp, empty_row_df])
    tmp.to_csv('zresults/%s - %s - %s - %s.csv' % (title, k_, res_, met_), index=False)
    print('zresults/%s - %s - %s - %s.csv' % (title, k_, res_, met_))
    return tmp