from ..nn.dnn.model.dnn import DNN 
from ..nn.dnn.model.noisednn import NoiseDNN
from PIML.util.constants import Constants

class BaseNN(Constants):
    def __init__(self, mtype):
        self.model =None
        self.mtype = mtype

    def set_model(self):
        if self.mtype == 'DNN':
            self.model = DNN()
        elif self.mtype == 'NoiseDNN':
            self.model = NoiseDNN()

    def set_model_shape(self, input_dim, output_dim):
        self.model.set_model_shape(input_dim, output_dim)
    
    def set_model_param(self, lr=0.01, dp=0.0, loss='mse', opt='adam', name=''):
        self.model.set_model_param(lr=lr, dp=dp, loss=loss, opt=opt, name=name)

        
    def prepare_DNN(self, input_dim, output_dim, lr=0.01, dp=0.0):
        self.model.set_model_shape(input_dim, output_dim)
        self.model.set_model_param(lr=lr, dp=dp, loss='mse', opt='adam', name='')
        self.model.build_model()

    def add_noise(self, x, noise):
        return x + noise


    def build_model(self, noise_level=None):
        if self.mtype == 'DNN':
            self.model.build_model()
        elif self.mtype == 'NoiseDNN':
            self.model.build_model(noise_level=noise_level)

