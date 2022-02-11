import os
import numpy as np
import pandas as pd
import logging
import h5py
import datetime
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as kl
import tensorflow.keras.regularizers as kr
import tensorflow.keras.activations as ka
from tensorflow.keras import Sequential as ks
from tensorflow.keras import optimizers as ko
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError

from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

import warnings
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
devices = tf.config.list_physical_devices('GPU')
for device in devices:
    tf.config.experimental.set_memory_growth(device, True)
logging.info(device)





class DNN(object):
    def __init__(self):
        # super(Autoencoder, self).__init__()
        # ------------model shape----------------
        self.input_dim = None
        self.output_dim = None
        self.hidden_dims = None
        self.input = None
        self.output = None
        self.units = None
        # ------------model param----------------
        self.model = None
        self.reg1 = None
        self.dp = None
        self.mtype = "DNN"
        self.callbacks=[]
        self.nn_rescaler = None
        self.name = None
        self.save_dir = None
        self.log_dir = None
        self.noise_level = None
        self.ep = None


    def set_model_shape(self, input_dim, output_dim, hidden_dims=[]):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input = keras.Input(shape = (self.input_dim, ), name='input')
        self.output = keras.Input(shape = (self.output_dim, ), name='out')
                
        self.hidden_dims = np.array(hidden_dims)
        self.units = self.get_units()

    def set_model_param(self, lr=0.1, dp=0.0, loss='mse', opt='adam'):
        self.lr = lr
        self.dp = dp
        self.opt = self.get_opt(opt)
        self.loss = loss
        # self.log_dir = "logs/fit/" + self.name        
        self.callbacks.append([
            EarlyStopping(monitor='loss', patience=40),
            ReduceLROnPlateau('loss',patience=8, min_lr=0.000001, factor=0.6),
        ])

    def set_tensorboard(self, log_dir, name="", verbose=1):
        self.name = self.get_model_name(name)
        log_path = os.path.join(log_dir, self.name)        
        self.callbacks.append([
            TensorBoard(log_dir=log_path, histogram_freq=verbose),
            ModelCheckpoint(self.name, save_best_only=True, monitor='val_loss', mode='min')
        ])
        return log_path


    def get_opt(self, opt):
        if opt == 'adam':
            return ko.Adam(learning_rate=self.lr, decay=1e-6)
        if opt == 'sgd':
            return ko.SGD(learning_rate=self.lr, momentum=0.9)
        else:
            raise 'optimizer not working'

    def get_model_name(self, name):
        out_name = f'{self.mtype[:3]}_nl{self.noise_level:.0f}_lr{self.lr}_I{self.input_dim}_h{len(self.hidden_dims)}_O{self.output_dim}_'
        if self.dp != 0:
            out_name = out_name + f'dp{self.dp}_'
        t = datetime.datetime.now().strftime("%d_%H%M")
        out_name = name + out_name + t
        return out_name.replace('.', '')

    def scale_predict(self, x_test):
        y_pred = self.model.predict(x_test)
        return self.nn_rescaler(y_pred)

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
            if self.input_dim <= 1000:
                hidden_dims = np.array([128, 64, 32, 16])
                # hidden_dims = np.array([512, 256, 128, 64, 32, 16])
                # hidden_dims = np.array([1024, 512, 256, 128, 64, 32, 16])
                # hidden_dims = np.array([2048, 1024, 512, 256, 128, 64, 32, 16])

                
                

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
            dp = 0 if ii == 0 else self.dp
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
                    kl.Dropout(dp_rate),
                    kl.LeakyReLU(),
                    kl.BatchNormalization(),
                    # keras.activations.tanh()
                    ])
        return layer

    def save_model(self):
        path = os.path.join(self.save_dir, self.name, 'model.h5')
        print("saving model to: ", path)
        self.model.save(path)
    
class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)