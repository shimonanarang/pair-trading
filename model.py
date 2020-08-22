
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import numpy as np
import pandas as pd	
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.regularizers import l2


def train(x,y):

    model = Sequential()
    model.add(LSTM(128,activation = 'relu', batch_input_shape =(x.shape[0],x.shape[1],x.shape[2]), dropout = 0.5))
    #model.add(LSTM(128,activation = 'relu', input_shape = (x.shape[0], 128)))
    model.add(Dense(100,activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss = 'mse', optimizer = 'adam')
    
    print("Model Summary: \n", model.summary())

    history = model.fit(x,y,epochs = 150, verbose = 1, shuffle = False, validation_split = 0.1)

    train_loss_values = history.history["loss"]
    val_loss = history.history["val_loss"]
    return val_loss, train_loss_values, model

