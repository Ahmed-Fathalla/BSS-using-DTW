import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from .ML.Dl_models import *
from .ML.exp_conf import df_cols, exp_dict
from .scoring.scoring import get_res
from .utils.time_utils import get_TimeStamp_str
from .utils.csv_utils import csv_create_empty_df, csv_append_row
from .extra_utils import *

results_cols=['exp', 'cluster', 'response','model','Response_min','Response_max', 'train_size', 'test_size',
              'MAE', 'RMSE', 'MSE', 'R2', 'MAPE',
              '-- tst_MAE ------', 'tst_RMSE', 'tst_MSE', 'tst_R2', 'tst_MAPE']

loop_dic = {'Trip_Duration_in_min':0, 'distance':1}

def Similarity_DL_exp(df, model_func, callbacks, fast_run=0,
                      batch_size=512, epochs=40, verbose=1,
                      start=0, scaling=0,
                      only_one=0,
                      cluster=-1):

    Respo_min = Respo_max = -100

    df = df[df_cols]

    if 'age_label_cust' in df.columns:
        df.pop('age_label_cust')
    if 'age_2_split' in df.columns:
        df.pop('age_2_split')

    i = 0

    f_res = []

    for response in ['Trip_Duration_in_min', 'distance']:
        i += 1
        timestamp = get_TimeStamp_str() + '-' + response[:3] + ' - '

        Respo_min, Respo_max = df[response].min(), df[response].max()
        d = df.copy()
        y = d.pop(response)
        x = d.copy()

        scaler = None
        if scaling:
            scaler = MinMaxScaler()
            y = scaler.fit_transform(y.values.reshape(-1, 1))

        model = 'NoModel----------'
        X_train = X_test = df.copy()

        if fast_run: x, y, epochs = x[:50], y[:50], 2
        X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=42)

        # ============================================
        if isinstance(model_func, list):
            model_str = model_func[loop_dic[response]]
            model = keras.models.load_model('models/'+model_str+'.keras')
        else:
            model, model_str = model_func(X_train.shape[1])
            # print( 'model_str = ' , model_str )
            model_str = timestamp+model_str
            model.save('models/%s.keras'%model_str )


        his = model.fit(X_train, y_train,
                        validation_split=0.10, verbose=verbose,
                        epochs=epochs, batch_size=batch_size,
                        callbacks=callbacks(verbose=verbose)
                        )

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)


        if scaling:
            y_train_pred, y_test_pred = scaler.inverse_transform(y_train_pred), scaler.inverse_transform(y_test_pred)
            y_train, y_test = scaler.inverse_transform(y_train), scaler.inverse_transform(y_test)

        s_train, results_train = get_res(y_train, y_train_pred)
        s_test, results_test = get_res(y_test, y_test_pred)
        results = ['exp_%-3d:' % i, cluster, response, model_str, Respo_min, Respo_max,
                   X_train.shape[0], X_test.shape[0], *results_train, *results_test]

        # results = ['exp_%-3d:' % i, cluster, response, '%s' % str(model)[:10], Respo_min, Respo_max,
        #            X_train.shape[0], X_test.shape[0], *results_train, *results_test]
        # ===============================================

        f_res.append(results)

    return f_res

def DL_Grid(dataframe, fast_run = 1,verbose=0, cluster=-1, tmp_str = ''):
    count = 0
    f_res = []
    res_str = 'GRU_grid - ' + get_TimeStamp_str() + '.csv'
    for gru_units in [16, 32, 64]:
        for nn_units in [16, 32, 64]:
            for i in range(3):
                count += 1
                # print('\n** %d --------------->>>>'%count, gru_units, nn_units, i)

                def get_GRU_model_(no_cols):
                    m = Sequential()
                    m.add(GRU(units=gru_units, activation='tanh', name='gru', return_sequences=False,
                              input_shape=(no_cols, 1)))
                    m.add(Dropout(0.2))
                    m.add(Dense(units=nn_units, activation='relu'))
                    m.add(Dropout(0.2))
                    m.add(Dense(units=nn_units / 2, activation='relu'))
                    m.add(Dense(units=1, activation='linear'))

                    m.compile(optimizer='adam',  # SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),
                              loss='mean_squared_error')
                    return m, '%d-%d-%d' % (gru_units, nn_units, i)


                print('\r', '\t** %d --------------->>>> %d %d -> %d  Shape:%d  Fast:%d'%
                      (count, gru_units, nn_units, i, dataframe.shape[0], fast_run), end='')

                results = Similarity_DL_exp(dataframe, get_GRU_model_,
                                            get_callBacks, verbose=verbose,
                                            fast_run=fast_run, scaling=1, epochs=40,
                                            cluster=cluster
                                           )
                f_res.extend(results)
                
                try:pd.DataFrame(np.array(f_res)).to_csv('clusters/%s - %s.csv'%(res_str,tmp_str), index=False)
                except:print('^&'*120)
                
                # try:
                    # pd.DataFrame(np.array(f_res), columns=results_cols).to_csv(res_str,index=False)
                    # print(res_str, '=======================')
                # except:
                    # print('\n\n Cannot dump it')
    return f_res

def GRU_clusters(best_models, verbose=0, fast_run=2000):
    from .utils.pkl_utils import load_pkl

    df = pd.read_csv('df_encoding2.csv')
    dic_df = pd.read_csv('dic.csv')
    res = load_pkl('clustering')

    cluster_df = None

    write_('output', '\n' * 5, '@' * 50)

    # modified_df = df[df['End Station Encoding'] != -1 ]
    # exp_res = Similarity_ML_exp(modified_df, start_model=1)
    # pd.DataFrame(np.array(exp_res), columns=results_cols).to_csv('Full results.csv', index=False)

    for i in range(2, 11):
        exp_df = None
        results = []
        tmp = dic_df.copy()
        tmp['cluster'] = res[str(i)]
        cluster_id_results = []
        for cluster_id in set(res[str(i)]):
            cluster_df = tmp[tmp['cluster'] == cluster_id]
            cluster_id_df = pd.DataFrame()
            for k in cluster_df['key'].values[:fast_run]:  # [:2] *************************************************************************
                w = [int(w) for w in k.split('-')]
                # print( 'w = ' , w )
                sub_df = get_data(df, w[0], w[1], return_type=0)
                cluster_id_df = pd.concat([cluster_id_df, sub_df])
            write_('GRU_Cluster_output', '\ncluster_df.shape ---------- cluster_id:%d' % cluster_id, cluster_df.shape,
                   "  DF length:", cluster_id_df.shape[0])

            # cluster_id_df[['Trip_Duration_in_min', 'distance']].to_csv('DS\%d - %d.csv'%(i,cluster_id), index=False)
            # exp_res = Similarity_ML_exp(cluster_id_df, cluster=cluster_id)
            exp_res = Similarity_DL_exp(cluster_id_df,
                                        best_models,
                                        get_callBacks,
                                        verbose=verbose,
                                        fast_run=0,
                                        scaling=1,
                                        epochs=40,
                                        cluster=cluster_id)

            cluster_id_results.extend(exp_res)
            # break  #    *************************************************************************

        all_res = pd.DataFrame(np.array(cluster_id_results), columns=results_cols)
        all_res.to_csv('clusters\GRU cluster %d.csv' % i, index=False)
        # break  # *************************************************************************

def GRU_clusters_random_weights( verbose=0,
                                 fast_run_cluster=2000,
                                 fast_run_model=0,
                                 cluster_start = 2,
                                 cluster_end = 11,
                                 key_start=0,
                                 key_end = 500):
    from .utils.pkl_utils import load_pkl
    print( 'key_start = ' , key_start )
    df = pd.read_csv('df_encoding2.csv')
    dic_df = pd.read_csv('dic.csv')
    res = load_pkl('clustering')

    cluster_df = None

    write_('output', '\n' * 5, '@' * 50)

    # modified_df = df[df['End Station Encoding'] != -1 ]
    # exp_res = Similarity_ML_exp(modified_df, start_model=1)
    # pd.DataFrame(np.array(exp_res), columns=results_cols).to_csv('Full results.csv', index=False)
    for i in range(cluster_start, cluster_end):
        exp_df = None
        results = []
        tmp = dic_df.copy()
        tmp['cluster'] = res[str(i)]
        cluster_id_results = []
        ll = list(set(res[str(i)]))
        print( 'll = ' , ll )
        for cluster_id in ll[key_start:key_end]:
            print( '-------------------- cluster_id = ' , cluster_id )
            cluster_df = tmp[tmp['cluster'] == cluster_id]
            cluster_id_df = pd.DataFrame()
            for k in cluster_df['key'].values[:fast_run_cluster]:  # [:2] *************************************************************************
                w = [int(w) for w in k.split('-')]
                # print( 'w = ' , w )
                sub_df = get_data(df, w[0], w[1], return_type=0)
                cluster_id_df = pd.concat([cluster_id_df, sub_df])
            write_('GRU_Cluster_output', '\ncluster_df.shape ---------- cluster_id:%d' % cluster_id, cluster_df.shape,
                   "  DF length:", cluster_id_df.shape[0])

            # cluster_id_df[['Trip_Duration_in_min', 'distance']].to_csv('DS\%d - %d.csv'%(i,cluster_id), index=False)
            # exp_res = Similarity_ML_exp(cluster_id_df, cluster=cluster_id)
            print('Cluster', i, '  ', cluster_id)
            exp_res = DL_Grid(cluster_id_df,
                              fast_run_model,
                              verbose,
                              cluster=cluster_id,
                              tmp_str = '%d %d'%(i, cluster_id))

            cluster_id_results.extend(exp_res)
            # break  #    *************************************************************************

            all_res = pd.DataFrame(np.array(cluster_id_results), columns=results_cols)
            all_res.to_csv('clusters/GRU cluster %d - %d.csv' % (i,cluster_id), index=False)
        # break