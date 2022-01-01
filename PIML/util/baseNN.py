from ..nn.dnn.model.dnn import DNN 
from PIML.util.constants import Constants

class BaseNN(Constants):
    def __init__(self):
        self.model =None

    def set_model(self, type='DNN'):
        if type == 'DNN':
            self.model = DNN()

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


