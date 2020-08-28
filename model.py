from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from sklearn.metrics import accuracy_score
from seq_norm import inverse_transform


def LSTM_train(x, y, batch_size):
    n_timesteps = x.shape[1]
    n_features = x.shape[2]
    model = Sequential()
    model.add(LSTM(128, activation='tanh', input_shape=(n_timesteps, n_features), return_sequences=True, dropout=0.5,
                   recurrent_dropout=0.0))
    model.add(LSTM(128, activation='tanh', return_sequences=True, dropout=0.8))
    model.add(LSTM(128, activation='tanh', return_sequences=True, dropout=0.8))
    model.add(LSTM(128, activation='tanh', return_sequences=True, dropout=0.8))
    model.add(LSTM(128, activation='tanh', return_sequences=True, dropout=0.8))
    model.add(LSTM(128, activation='tanh', return_sequences=True, dropout=0.8))
    model.add(LSTM(128, activation='tanh', return_sequences=True, dropout=0.8))
    model.add(Flatten())
    model.add(Dense(100, activation='tanh'))
    model.add(Dropout(0.8))
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(0.8))
    model.add(Dense(1))
    opt = Adam(learning_rate=1e-5)
    model.compile(loss='mae', optimizer=opt)
    print("---------------------------- 6")
    callbacks_list = [
        ModelCheckpoint(
            filepath="LSTM-weights-best.hdf5",
            monitor='val_loss',
            save_best_only=True,
            mode='auto',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=10,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1
        )

    ]
    # y = np.expand_dims(y, 1)
    # history = model.fit(x,y,epochs = 30, verbose = 1, validation_split = 0.2, shuffle = False)
    history = model.fit(x, y, epochs=150, batch_size=batch_size, verbose=1, validation_split=0.2,
                        callbacks=callbacks_list, shuffle=False)
    train_loss_values = history.history["loss"]
    val_loss = history.history["val_loss"]
    return train_loss_values, val_loss, model


def prediction(x, model, norm_y):
    prediction = model.predict(x)

    # using inverse trasnform function from window normalization
    prediction = inverse_transform(prediction, norm_y)
    return prediction


'''ReduceLROnPlateau(
        monitor = 'val_loss',
        factor = 0.1,
        patience=3,
        verbose=1
        ),
        EarlyStopping(
            monitor = 'val_loss',
            patience=10, 
            verbose = 1
        )'''






