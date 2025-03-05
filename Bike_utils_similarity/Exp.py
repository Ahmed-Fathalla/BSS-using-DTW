from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from .ML.Dl_models import *
from .ML.exp_conf import df_cols, responses_variables, results_cols
from .scoring.scoring import get_res
from .utils.time_utils import get_TimeStamp_str
from .utils.csv_utils import csv_create_empty_df, csv_append_row
from .extra_utils import *

def ML_exp(df, start_model=0, end_model=-1 , cluster=-1, target=-1):
    from .ML.ML_models import model_lst
    timestamp = get_TimeStamp_str()
    Respo_min = Respo_max = -100

    df = df[df_cols]

    i = 0

    f_res = []

    target = responses_variables if target==-1 else [responses_variables[target]]

    for response in target:
        Respo_min, Respo_max = df[response].min(), df[response].max()
        d = df.copy()
        # d = d[:60]
        y = d.pop(response)

        for ccol in responses_variables:
            try:
                d.pop(ccol)
                print('removing duration')
            except:
                ...

        x = d.copy()
        model = 'NoModel----------'
        X_train = X_test = df.copy()
        X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2,
                                                            random_state=42)
        for model in model_lst[start_model:end_model]:
            i += 1
            
            print(cluster, '==>', response, '=>', model)

            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            s_train, results_train = get_res(y_train, y_train_pred)

            s_test, results_test = get_res(y_test, y_test_pred)

            results = ['exp_%-3d:' % i, cluster, response, '%s' % str(model)[:10], Respo_min, Respo_max,
                       X_train.shape[0], X_test.shape[0], *results_train, *results_test]
            # print( 'results = ' , results )

            f_res.append(results)

    return f_res

# def run_all_data(df, model_func, callbacks, fast_run=0, batch_size= 512, epochs=20, verbose=1, start=0):
def run_all_data(df, model_func, callbacks, fast_run=0, batch_size= 1024, epochs=20, verbose=1, start=0, scaling=0):
    df = df[ df_cols ] # , 'age_label_cust'
    Respo_min = Respo_max = -100
    if fast_run: output_file = 'z/Fast - Compete df - %s.csv'% get_TimeStamp_str()
    else:output_file = 'z/Compete df - %s.csv'% get_TimeStamp_str()
    print('\n', output_file, '\n')
    csv_create_empty_df(output_file, results_cols)

    k, itm = '', '' # newly added ---------------------------------------------------------
    dd = df.copy()  # newly added ---------------------------------------------------------
    if 'age_label_cust' in dd.columns:
        dd.pop('age_label_cust')

    i = 0
    for response in responses_variables:
        print('All -', i)
        i += 1
        if i < (start + 1):
            continue

        Respo_min , Respo_max = dd[response].min(), dd[response].max()
        d = dd.copy()
        y = d.pop(response)
        x = d.copy()
        X_train= X_test = dd.copy()

        scaler = None
        if scaling:
            scaler = MinMaxScaler()
            y = scaler.fit_transform(y.values.reshape(-1, 1))

        try:
            if fast_run: x, y, epochs = x[:50], y[:50], 2
            X_train, X_test, y_train, y_test = train_test_split(x, y,train_size=0.8, test_size = 0.2, random_state = 42)
            model = 'NoModel----------'
            model = model_func(X_train.shape[1])
            s = '\n'*2+'exp_%-3d:'%i+'%-20s'%k+' %-20s'%itm+ ' %-20s'%response[:14]+ ' %-10s'%str(model)[:6]+ '     train: '+str(len(X_train))+ ' test: '+str(len(X_test)) + '\n'
            his=model.fit(X_train, y_train,
                        validation_split = 0.10,verbose=verbose,
                        epochs=epochs, batch_size= batch_size,
                        callbacks= callbacks(verbose)
                        )

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            if scaling:
                y_train_pred, y_test_pred = scaler.inverse_transform(y_train_pred), scaler.inverse_transform(y_test_pred)
                y_train, y_test = scaler.inverse_transform(y_train), scaler.inverse_transform(y_test)

            s_train, results_train=get_res(y_train, y_train_pred)
            s_test, results_test=get_res(y_test, y_test_pred)
            results = ['exp_%-3d:'%i, k, itm, response,  '%s'%str(model)[:10], Respo_min , Respo_max,
                        X_train.shape[0],X_test.shape[0],*results_train, *results_test]
            csv_append_row(output_file, results)
            s+='\n'*2

        except Exception as exc:
            with open('%s.txt'%output_file, 'a') as myfile:
                myfile.write( 'Error -> ' +'exp_%-3d:'%i+'%-20s'%k+' %-20s'%itm+ ' %-20s'%response[:14]+ ' %-10s'%str(model)[:6]+ '     train: '+str(len(X_train))+ ' test: '+str(len(X_test)) + '\n' + str(exc)+'\n\n')
                print( 'Error -> ' +'%-3d:'%i+'%-20s'%k+' %-20s'%itm+ ' %-20s'%response[:14]+ ' %-10s'%str(model)[:6]+ '     train: '+str(len(X_train))+ ' test: '+str(len(X_test)) + '\n' + str(exc)+'\n\n')
        csv_append_row(output_file, ['' for _ in range(len(results_cols))])

