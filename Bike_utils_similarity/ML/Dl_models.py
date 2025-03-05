import numpy as np
import pandas as pd

from tensorflow import keras # for building Neural Networks
from keras.models import Sequential # for creating a linear stack of layers for our Neural Network
from keras import Input # for instantiating a keras tensor
from keras.layers import Bidirectional, GRU, RepeatVector, Dense, Dropout # for creating layers inside the Neural Network
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

from ..utils.time_utils import get_TimeStamp_str

def get_callBacks(verbose):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=3)
    cp = ModelCheckpoint('models/z_best_model - %s.h5'%(get_TimeStamp_str()), monitor='val_loss', mode='min', verbose=verbose, save_best_only=True)
    return [es, cp]

def save_csv(test, pred, file):
    df = pd.DataFrame()
    df['true'] = list(test.reshape(-1, 1).flatten().tolist())
    df['pred'] = list(pred.reshape(-1, 1).flatten().tolist())
    df['residual'] = np.abs(df['true'] - df['pred'])
    df.to_csv('csv/%s_%s.csv' % (file, get_TimeStamp_str()), index=False)
    print('csv/%s_%s.csv' % (file, get_TimeStamp_str()))

# https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
# https://www.kaggle.com/code/thebrownviking20/intro-to-recurrent-neural-networks-lstm-gru
def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))

    return LearningRateScheduler(schedule)



# def get_GRU_model(no_cols):
    # m = Sequential()
    # m.add(GRU(units=50, return_sequences=True, input_shape=(no_cols,1), activation='tanh'))
    # m.add(Dropout(0.2))
    # m.add(GRU(units=50, return_sequences=True, input_shape=(no_cols,1), activation='tanh'))
    # m.add(Dropout(0.2))
    # m.add(GRU(units=50, return_sequences=True, input_shape=(no_cols,1), activation='tanh'))
    # m.add(Dropout(0.2))
    # m.add(GRU(units=50, activation='tanh'))
    # m.add(Dropout(0.2))
    # m.add(Dense(units=1))

    # m.compile(optimizer='adam', # SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),
              # loss='mean_squared_error')

    # return m

