from .dnn import DNN
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
        callbacks_noise = NoiseAug()
    
    # def get_noise_model(self, noise_level=1):
    #     inputs = [self.input, self.input2]
    #     noise_layer = NoiseAugLayer(self.input_dim, noise_level)
    #     x = noise_layer(inputs)
    #     model = keras.Model(inputs=inputs, outputs=x, name='noisednn')
    #     return model

    def get_model(self):
        x = self.input
        model = keras.Model(inputs=self.input, outputs=x, name='dnn')
        return model




    def fit(self, x_train, std_train, y_train, ep=1, batch=512, verbose=2):
        self.model.fit([x_train, std_train], y_train, 
                        epochs=ep, batch_size=batch,  validation_split=0.2, 
                        shuffle=True, verbose=verbose, callbacks=self.callbacks)


class NoiseAug(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.layers[0].stddev = 100
        print('updating sttdev in training')
        print(self.model.layers[0].stddev)


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