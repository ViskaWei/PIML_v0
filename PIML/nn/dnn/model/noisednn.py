from .dnn import DNN
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

class NoiseDNN(DNN):
    def __init__(self):
        super().__init__()
        self.input2 = keras.Input(shape=(self.input_dim, ), name='stddev')
        # self.

    def build_dnn(self, noise_level=1):
        self.model = self.get_model()
    
    def get_model(self):
        x = self.input
        model = keras.Model(inputs=self.input, outputs=x, name='dnn')
        return model

    def build_DataGenerator(self, x_train, x_std, y_train, batch_size=32, shuffle=True, validation_split=0.2):
        cut = int(x_train.shape[0] * (1 - validation_split))
        print(f"cut: {cut}")
        training_generator = DataGenerator(x_train[:cut], x_std[:cut], y_train[:cut], batch_size=batch_size, shuffle=shuffle)
        validation_generator = DataGenerator(x_train[cut:], x_std[cut:], y_train[cut:], batch_size=batch_size, shuffle=shuffle)
        return training_generator, validation_generator

    def fit(self,x_train, x_std, y_train, ep=1, batch=512, verbose=2, shuffle=True):
        training_generator, validation_generator = self.build_DataGenerator(x_train, x_std, y_train, batch_size=batch, shuffle=shuffle, validation_split=0.2)        
        self.model.fit_generator(generator=training_generator,
                                validation_data=validation_generator,
                                use_multiprocessing=False,
                                workers=6)


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, data_std, labels, noise_level=1, batch_size=32, num_classes=None, shuffle=True):
        self.batch_size = batch_size
        self.data = data
        self.data_std = self.load_data_std(data_std)
        self.nData, self.nDim = data.shape
        self.indices = np.arange(self.nData)
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.labels = labels
        self.nLabels = labels.shape[1]
        self.noise_level = noise_level
        self.on_epoch_end()

    def load_data_std(self, data_std):
        if data_std.any() < 0:
            raise ValueError("data_std must be positive")
        else:
            return data_std

    def __len__(self):
        return self.nData // self.batch_size

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]
        
        X, y = self.__get_data(batch)
        print("X=", X, "y=", y)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(self.nData)
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        X = np.empty((self.batch_size, self.nDim))
        y = np.empty((self.batch_size, self.nLabels), dtype=int)
        
        for i, id in enumerate(batch):
            noise = np.random.normal(0, self.noise_level * self.data_std[i], (self.batch_sizem, self.nDim))
            X[i,] = self.data[id] + noise
            y[i] = self.labels[id]

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