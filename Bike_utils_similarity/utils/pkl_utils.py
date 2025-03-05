import pickle

def load_pkl(file_name):
    path = file_name if str(file_name).endswith('.pkl') else '%s.pkl'%file_name
    with open(path, "rb") as f:
        return pickle.load(f)

def dump(data, path):
    try:
        with open(path+'.pkl', "wb") as f:
            pickle.dump(data, f)
        # print('successfully dump to "%s.pkl"'% path)
    except Exception as exc:
        print('\n**** Err:\n', traceback.format_exc())
