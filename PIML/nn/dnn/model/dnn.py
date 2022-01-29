import os
import numpy as np
import pandas as pd
import logging
# import signal
import h5py
import datetime
# from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as kl
import tensorflow.keras.regularizers as kr
import tensorflow.keras.activations as ka
from tensorflow.keras import Sequential as ks
from tensorflow.keras import optimizers as ko
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError

from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
# import tensorflow.python.keras.optimizer_v2 as ko



class DNN(object):
    def __init__(self):
        # super(Autoencoder, self).__init__()
        # model shape
        self.input_dim = None
        self.output_dim = None
        self.hidden_dims = None
        self.input = None
        self.output = None
        self.units = None
        # model param
        self.model = None
        self.reg1 = None
        self.dp = None
        self.mtype = "DNN"
        self.callbacks=[]
        self.nn_rescaler = None

    def set_model_shape(self, input_dim, output_dim, hidden_dims=[]):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input = keras.Input(shape = (self.input_dim, ), name='input')
        self.output = keras.Input(shape = (self.output_dim, ), name='out')
                
        self.hidden_dims = np.array(hidden_dims)
        self.units = self.get_units()

    def set_model_param(self, lr=0.01, dp=0.0, loss='mse', opt='adam', name=''):
        self.lr = lr
        self.dp = dp
        self.opt = self.get_opt(opt)
        self.loss = loss
        # self.name = self.get_name(name)
        # self.log_dir = "logs/fit/" + self.name        
        self.callbacks.append([
            EarlyStopping(monitor='loss', patience=10),
            ReduceLROnPlateau('loss',patience=10, min_lr=0., factor=0.1),
            # TensorBoard(log_dir=self.log_dir)
            # TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        ])

    # def set_train_param(self, ep=50, batch=512, verbose=2):
    #     self.ep = ep
    #     self.batch = batch
    #     self.verbose = verbose

    def get_opt(self, opt):
        if opt == 'adam':
            return ko.Adam(learning_rate=self.lr, decay=1e-6)
        if opt == 'sgd':
            return ko.SGD(learning_rate=self.lr, momentum=0.9)
        else:
            raise 'optimizer not working'

    # def get_name(self, name):
    #     lr_name = -np.log10(self.lr)
    #     out_name = f'{self.loss}_lr{lr_name}_h{len(self.hidden_dims)}'
    #     if self.dp != 0:
    #         out_name = out_name + f'dp{self.dp}_'
    #     t = datetime.datetime.now().strftime("%m%d-%H%M%S")
    #     out_name = out_name + name + '_' + t
    #     return out_name.replace('.', '')


    # def run(self, x_train, y_train, x_test, y_test, )

    def predict(self, x_test, scaler=None):
        y_pred = self.model.predict(x_test)
        if scaler is None:
            return y_pred
        else:
            return scaler(y_pred)

    def fit(self, x_train, y_train, nEpoch=50, batch=512, verbose=2):
        self.model.fit(x_train, y_train, 
                    epochs=nEpoch, 
                    batch_size=batch, 
                    validation_split=0.2, 
                    callbacks=self.callbacks,
                    shuffle=True,
                    verbose=verbose
                    )
        if verbose == 0:
            prints=f"| EP {nEpoch} |"
            for key, value in self.model.history.history.items():
                prints = prints +  f"{key[:5]}: {value[-1]:.4f} | "
            print(prints)
        tf.keras.backend.clear_session()
            # print(self.model.summary())
    
    def log2(self,r):
        x = self.input_dim // r
        return int(2**np.floor(np.log2(x)))

    def get_units(self):
        if self.hidden_dims.size == 0:
            if self.input_dim <= 150:
                hidden_dims = np.array([64, 32, 16])

                # hidden_dims = np.array([128, 64, 32])
            elif self.input_dim < 2048:
                hidden_dims = np.array([1024, 512, 128, 32])
            else:
                hidden_dims = np.array([self.log2(2), self.log2(4), self.log2(8)])
            self.hidden_dims = hidden_dims
        self.hidden_dims = self.hidden_dims[self.hidden_dims > self.output_dim]
        units = [self.input_dim, *self.hidden_dims, self.output_dim]
        print(f"Layers: {units}")
        return units 

    def build_dnn(self, **args):
        x = self.input
        for ii, unit in enumerate(self.units[1:]):
            name = 'l' + str(ii)
            x = self.add_dense_layer(unit, dp_rate=self.dp, reg1=self.reg1, name=name)(x)
        self.model = keras.Model(self.input, x, name="dnn")

    

    def build_model(self, **args):
        self.build_dnn(**args)
        self.model.compile(
                loss=self.loss,
                optimizer=self.opt,
                # metrics=['acc'],
                metrics=[MeanSquaredError()]
            )

    def add_dense_layer(self, unit, dp_rate=0., reg1=None, name=None):
        if reg1 is not None:
            kl1 = tf.keras.regularizers.l1(reg1)
        else:
            kl1 = None

        layer = ks([kl.Dense(unit, kernel_regularizer=kl1, name=name),
                    # kl.BatchNormalization(),
                    kl.LeakyReLU(),
                    kl.Dropout(dp_rate)
                    # keras.activations.tanh()
                    ])
        return layer

    # def eval(self, x_test, y_test, WR, Prng):
    #     if self.top is not None:
    #         if self.mtype == "PCA":
    #             y_pred = self.model.predict(x_test[:,:self.top])
    #         elif self.mtype == "PCP":
    #             x_test = self.pcpflux_top(x_test, top=(self.top // 4))
    #             y_pred = self.model.predict(x_test)
    #     self.MSE = np.mean(np.square(y_test - y_pred), axis=0)
    #     self.MAE = np.mean(np.abs(y_test - y_pred), axis=0)
    #     self.RMS = np.sqrt(self.MSE)
    #     self.MAEP = np.multiply(self.MAE, Prng)[0]
    #     self.plot_pred(y_test, y_pred, WR=WR)
    #     return y_pred

# class MyCallback(Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         if epoch % 10 == 1:
#             print(epoch, self.model.losses)
    # def plot_pred(self, y_test, y_pred, WR=""):
    #     f, axs = plt.subplots(1,self.output_dim, figsize=(5*self.output_dim,4), sharex="row", sharey="row", facecolor="w")
    #     for pdx in range(self.output_dim):
    #         ax = axs[pdx]
    #         ax.scatter(y_test[:,pdx], y_pred[:,pdx], s=1, c=y_test[:,pdx])
    #         ax.plot([[0,0], [1,1]], "r")
    #         ax.annotate(f"\n{WR}\nMSE={self.MSE[pdx]:.4f}\n$\Delta$ {self.pname[pdx]}={self.MAEP[pdx]:.2f}\nRMS={self.RMS[pdx]:.2f}", 
    #                         (0.15, 0.75), xycoords="axes fraction")
    #         ax.set_xlabel(f"Norm {self.pname[pdx]}")
    #     axs[0].set_ylabel(f"Top {self.top} {self.mtype} Pred")    
