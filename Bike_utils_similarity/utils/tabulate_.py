from tabulate import tabulate
import pandas as pd
import numpy as np

def df_to_file(df=None, data_=None, round_=5, file=None, padding='left', rep_newlines='\t', print_=True, wide_col='', pre='', post='',
                return_str=True):
    if df is None:
        a = data_[0]
        if isinstance(data_[0], list):
            a = np.array(a).reshape(1, -1)
        df = pd.DataFrame(a, columns = data_[1])
    headers = [wide_col+str(i)+wide_col for i in df.columns.values]
    c = rep_newlines + tabulate(df.round(round_).values,
                                headers=headers,
                                stralign=padding,
                                disable_numparse=1,
                                tablefmt = 'grid' # 'fancy_grid' ,
                                ).replace('\n', '\n'+rep_newlines)
    if print_:print(c)
    
    if file is not None:
        with open(file, 'a', encoding="utf-8") as myfile:
            myfile.write( "\n" + pre + c + post + '\n\n')
    
    # print( 'return_str = ' , return_str )
            
    if return_str:
        return c




### =================================================================
###                           From Oil work
### =================================================================
###       D:\D_ Study\Big Data\Python Code\5 Oil\utils\Callbacks\callback_utils.py

# def df_to_str(df, headers, file=None, padding='left', rep_newlines='\t', wide_col='', print_=False):
#     c = rep_newlines + tabulate(
#                                 df,
#                                 headers=headers,
#                                 stralign=padding,
#                                 disable_numparse=1,
#                                 tablefmt = 'grid' # 'fancy_grid' ,
#                                 ).replace('\n', '\n'+rep_newlines)
#
#     if file is not None:
#         with open(file, 'a') as myfile:
#             myfile.write(  c )
#
#     if print_:
#         print(c)
#
#     return c