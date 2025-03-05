import csv, operator
import pandas as pd

def csv_create_empty_df(file, cols):
    df = pd.DataFrame(data=None, columns=cols)
    df.to_csv('%s'%file, index=False)

def csv_append_row(file, row):
    with open('%s'%file, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)