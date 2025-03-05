import numpy as np
import pandas as pd

def write_(file, *s):
    s_=''
    for i in s:
        s_ += str(i) + ' '
    with open(file+'.txt', 'a') as myfile:
        myfile.write( s_ )
    print(s_)

class data_partition:
    def __init__(self, dataframe):
        self.data_frame = dataframe

    def get_data(self, start_, end_, info_='data'):
        if info_ == 'shape':
            return self.data_frame[(
                    (self.data_frame['Start Station Encoding'] == start_) & (self.data_frame['End Station Encoding'] == end_) |
                    (self.data_frame['Start Station Encoding'] == end_) & (self.data_frame['End Station Encoding'] == start_)

            )].shape[0]
        elif info_ == 'data':
            return self.data_frame[(
                    (self.data_frame['Start Station Encoding'] == start_) & (self.data_frame['End Station Encoding'] == end_) |
                    (self.data_frame['Start Station Encoding'] == end_) & (self.data_frame['End Station Encoding'] == start_)
            )]


def empty_row(columns, rows=1):
    a = pd.DataFrame(np.array([np.nan]*len(columns)).reshape(1,-1), columns=columns)
    return pd.concat([a,pd.DataFrame(np.array(['' for _ in range(rows - 1)]))])




