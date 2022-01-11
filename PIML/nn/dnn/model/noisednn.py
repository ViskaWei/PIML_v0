from .dnn import DNN
from tensorflow.keras.callbacks import Callback

class NoiseDNN(DNN):
    def __init__(self):
        super().__init__()



    
class NoisePCA(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.layers[0].stddev = 100
        print('updating sttdev in training')
        print(self.model.layers[0].stddev)
cc = MyCustomCallback()