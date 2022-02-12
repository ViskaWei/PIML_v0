import logging
from ast import Constant
from .dnn import DNN
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

class NzDNN(DNN):
    def __init__(self):
        super().__init__()
        self.input2 = keras.Input(shape=(self.input_dim, ), name='stddev')
        self.eigv = None
        self.noise_level = None
        self.mtype = 'nzDNN'
        self.nn_scaler = None
        self.nn_rescaler = None


    def build_DataGenerator(self, x_train, x_std, y_train, batch_size=32, shuffle=True, validation_split=0.2):
        cut = int(x_train.shape[0] * (1 - validation_split))
        # print(f"cut: {cut}")
        training_generator = DataGenerator(x_train[:cut], x_std[:cut], y_train[:cut], eigv=self.eigv, batch_size=batch_size, shuffle=shuffle)
        validation_generator = DataGenerator(x_train[cut:], x_std[cut:], y_train[cut:], eigv=self.eigv, batch_size=batch_size, shuffle=shuffle)
        return training_generator, validation_generator

    def fit(self,x_train, y_train, nEpoch=1, batch=512, verbose=2, shuffle=True):
        x_data, x_std = x_train
        assert (x_std.any() >= 0)
        x_std_nz = self.noise_level * x_std
        training_generator, validation_generator = self.build_DataGenerator(x_data, x_std_nz, y_train, 
                            batch_size=batch, shuffle=shuffle, validation_split=0.2)        
        self.model.fit(training_generator, 
                        validation_data=validation_generator, 
                        callbacks=self.callbacks,
                        batch_size=batch, 
                        shuffle=shuffle, 
                        epochs=nEpoch, 
                        verbose=verbose)
        self.finish(nEpoch, verbose)




class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, data_std, labels, eigv=None, batch_size=256, shuffle=True):
        self.data = data
        self.data_std = data_std
        self.nData, self.nDim = data.shape
        self.nStd = data_std.shape[1]
        self.indices = np.arange(self.nData)
        self.labels = labels
        self.eigv = eigv
        self.batch_size = batch_size
        self.on_epoch_end(shuffle)

    def __len__(self):
        return self.nData // self.batch_size

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]
        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self, shuffle=True):
        self.index = np.arange(self.nData)
        if shuffle == True:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        X = self.data[batch]
        y = self.labels[batch]
        std = self.data_std[batch]
        noise = np.random.normal(0, std, size=(self.batch_size, self.nStd))
        if self.eigv is not None:
            noise = noise.dot(self.eigv.T)
        X = X + noise
        return X, y






# class NoiseAug(Callback):
#     def on_epoch_begin(self, epoch, logs=None):
#         self.model.layers[0].stddev = 100
#         print('updating sttdev in training')
#         print(self.model.layers[0].stddev)


# #Noise augmentation layer -------------------------------------------------
# class NoiseAugLayer(tf.keras.layers.Layer):
#     def __init__(self, num_outputs, noise_level=1.0):
#         super(NoiseAugLayer, self).__init__()
#         self.num_outputs = num_outputs
#         self.noise_level = noise_level

#     def get_noise(self, stddev):
#         print(stddev)
#         noise = tf.random.normal(shape=(self.num_outputs, ), mean=0, stddev=stddev, seed=42)
#         return noise
        
#     def call(self, inputs):
#         x, stddev = inputs
#         if (self.noise_level is not None): stddev = stddev * self.noise_level
#         # noise = self.get_noise(stddev)
#         noise = stddev
#         return x + noise

# #Noise augmentation layer -------------------------------------------------
# class NoiseAugLayer(tf.keras.layers.Layer):
#     def __init__(self, num_outpus, std_fn=None, noise_level=1.0):
#         super(NoiseAugLayer, self).__init__()
#         self.std_fn = self.get_std_fn(std_fn)
#         self.num_outputs = num_outpus
#         self.noise_level = noise_level

#     def get_std_fn(self, std_fn):
#         # stddev = tf.math.sqrt(var)
#         if std_fn is None:
#             return lambda x: x

#     def get_noise(self, stddev):
#         print(stddev)
#         noise = tf.random.normal(shape=(1,self.num_outputs), mean=0, stddev=stddev, seed=42)
#         return noise
        
#     def call(self, inputs):
#         stddev = self.std_fn(inputs)
#         if self.noise_level >= 1.0: stddev = stddev * self.noise_level
#         noise = self.get_noise(stddev)
#         return inputs + noise

    
# class NoisePCA(Callback):
#     def on_epoch_begin(self, epoch, logs=None):
#         self.model.layers[0].stddev = 100
#         print('updating sttdev in training')
#         print(self.model.layers[0].stddev)
# cc = MyCustomCallback()