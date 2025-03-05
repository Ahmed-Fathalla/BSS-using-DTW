import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as MAE,\
                            mean_squared_error as MSE,\
                            r2_score as R2
from ..utils.tabulate_ import df_to_file
from sklearn.metrics._regression import _check_reg_targets

def MAPE_Other(y_true, y_pred, multioutput='uniform_average'):
    '''
    This function is implemented by: Ahmed Fathalla <a.fathalla@science.suez.edu.eg>,
         The implementation of MAPE function follows sklearn/metrics regression metrics
    Parameters
    ----------
        y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
            Ground truth (correct) target values.

        y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
            Estimated target values.
    '''
    _, y_true, y_pred, _ = _check_reg_targets(y_true, y_pred, multioutput)
    y_true[y_true == 0.0] =  0.000001
    assert not(0.0 in y_true), 'MAPE arrises an Error, cannot calculate MAPE while y_true has 0 element(s). Check \"utils\_scoring_metrics.MAPE"'
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def RMSE(y_true, y_pred):
    return np.sqrt(MSE(y_true, y_pred))

metric_lst = [MAE, RMSE, MSE, R2, MAPE_Other][:]
metric_names = ['MAE', 'MSE', 'RMSE', 'R2', 'MAPE'][:len(metric_lst)]
def get_res(y_true, y_pred, round_=7,metric_lst=metric_lst, get_res_df = True, file=None):
    # y_true = np.array(y_true)
    # y_pred = np.array(y_pred)
    results = []
    for metric in metric_lst:
        results.append( round(metric( y_true=y_true, y_pred=y_pred),round_) )

    # if get_res_df:
    df = df_to_file(data_=(results, metric_names ), file=file, print_=False)
    # print('=======================', df)
    return df, results

# metric_lst = [MAE, MSE, RMSE, R2]
# metric_names = ['MAE', 'MSE', 'RMSE', 'R2']
# def get_res(y_true, y_pred, round_=7,metric_lst=metric_lst, get_res_df = True):
#     # y_true = np.array(y_true)
#     # y_pred = np.array(y_pred)
#     results = []
#     for metric in metric_lst:
#         results.append( round(metric( y_true=y_true, y_pred=y_pred),round_) )
#
#     if get_res_df:
#         df = df_to_file(data_=(results, metric_names ))
#     return results