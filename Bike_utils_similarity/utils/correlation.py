'''
get correlation and p-values

https://www.youtube.com/watch?v=TFqXnafVChI&list=PLmJ41iPm38bgll15Fj1Eu4ApAsjlWeh2j&index=13
check the colinearity using Variance Inflation Factor (VIF), if the data has high colinearity, then use PCA to remove the colinearity
'''
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.formula.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
import matplotlib.pyplot as plt
import seaborn as sns

def get_corr(a,b):
    pearsonr(a,b)[0]
    
def get_VIF(df, show_feat_name=True, return_df=False):
    object_feature = [i for i,j in zip(df.dtypes.index, df.dtypes) if j=='object']
    if len(object_feature) > 0:
        from .feature_encoding import custom_le
        for col in object_feature:
            df.loc[:,col] = custom_le(df[col]).values
    vif_series = pd.Series([vif(df.values, i) for i in range(df.shape[1])], index=df.columns).reset_index()
    vif_series.columns = ['Feature_name', 'VIF']
    if show_feat_name:
        print(vif_series.set_index('Feature_name'))
    else:
        vif_series.index = range(1,df.shape[1]+1)
        print(vif_series[['VIF']])
        
    return df if return_df else None 

def get_corr_matrix(df, target, collinear_threshold=0.07, show_feat_name=True, get_vif=False, return_df=False, plt_heatmap=True):
    '''
    df: dataframe of features with/without the response variable
    y: name or series of the response variable
    '''
    def pdng(str_, i, padding_char = ' '):
        return str_+ padding_char*(i-len(str_))
    def get_max_len(lst):
        max_len = 0
        for col in lst:
            if len(str(col)) > max_len:
                max_len = len(str(col))
        return max_len
    max_len = get_max_len(df.columns)+2
    
    df = df.copy()
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df, columns = range(1,df.shape[1]+1))
    feat_cols = list(df.columns)   
    if isinstance(target,str): # if y is the str-name of the response variable 
        feat_cols.remove(target)
    elif isinstance(target,pd.Series): # if y is the pd.Series of the response variable 
        df[target.name] = target.values
        target = target.name
    
    df = df.loc[:,[ target ,*feat_cols ]].copy()
    feat_dic = dict(zip(range(1, len(feat_cols)+1), feat_cols))
    def print_feature(i):
        if show_feat_name:return pdng(feat_dic[i], max_len) #'%-20s'%feat_dic[i]
        else:return '%-2d'%i
    
    # Encode object features with label encoder
    object_feature = [i for i,j in zip(df.dtypes.index, df.dtypes) if j=='object']
    if len(object_feature) > 0:
        from .feature_encoding import custom_le
        for col in object_feature:
            df.loc[:,col] = custom_le(df[col]).values
    
    corr_matrix = df.corr()
    if plt_heatmap:
        # plt.rcParams["figure.figsize"] = [df.shape[1]*2,df.shape[1]*2]
        sns.heatmap(corr_matrix, cmap="YlGnBu", annot=True)
        plt.savefig( 'Corr.pdf', bbox_inches='tight' )  
        plt.show()
        
    print('\n\n' + 'Features that have multiColinearity more than %-.2f\n'%collinear_threshold+ '-'*55)
    corr_matrix = corr_matrix.values
    for i in range(2,corr_matrix.shape[0]):
        for j in range(1, i):
            if np.abs(corr_matrix[i,j])>collinear_threshold:
                print('%s %s'%(print_feature(i),print_feature(j)), corr_matrix[i,j])
    print('='*60+'\n\n' + 'Corr with the Response variable\n'+ '-'*31)
    # tt = df.corr()[target].reset_index()
    # tt.columns = ['Feature_name', 'VIF']
    for j in range(1, corr_matrix.shape[0]):
        print('%s'%print_feature(j), corr_matrix[j,0])
    if get_vif:
        print('='*60,'\n\nVariance Inflation Factor\n'+ '-'*25)
        return get_VIF(df.loc[:,feat_cols], show_feat_name, return_df)

def p_value_Elimination(x, Y, columns, sl=0.05):
    '''
    https://towardsdatascience.com/feature-selection-correlation-and-p-value-da8921bfb3cf
    Selecting columns based on p-value

    '''
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    columns = np.delete(columns, j)
                    
    regressor_OLS.summary()
    return x, columns
# data_modeled, selected_columns = p_value_Elimination(data.iloc[:,1:].values, data.iloc[:,0].values, SL, selected_columns)    