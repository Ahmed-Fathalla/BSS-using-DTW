from ..scoring.scoring import get_res
from ..utils.time_utils import get_TimeStamp_str
from ..utils.csv_utils import csv_create_empty_df, csv_append_row

metrics_ = ['tst_MAE', 'tst_RMSE', 'tst_MSE', 'tst_R2', 'tst_MAPE']

responses_variables = ['Trip_Duration_in_min', 'distance']

cat_features =  ['User Type', 'Start Station Encoding', 
                 'DO_week', 'DO_month', 'month',
                 'year', 'hour', 'min']

df_cols =  [*cat_features,'Trip_Duration_in_min', 'distance', 'age', 'Gender']

results_cols=['exp', 'cluster','response','model','Response_min','Response_max', 'train_size', 'test_size',
              'MAE', 'RMSE', 'MSE', 'R2', 'MAPE',
              '-- tst_MAE ------', 'tst_RMSE', 'tst_MSE', 'tst_R2', 'tst_MAPE']

def write_(file, s_):
    with open(file, 'a') as myfile:
        myfile.write( s_ )
    print(s_)


