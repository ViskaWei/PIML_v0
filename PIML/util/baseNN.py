import os
from ..nn.dnn.model.dnn import DNN 
from ..nn.dnn.model.nzdnn import NzDNN
from PIML.util.constants import Constants

class BaseNN(Constants):
    def __init__(self,):
        self.cls =None
        self.mtype = None
        self.log_dir = Constants.LOG_DIR
        self.save_dir = Constants.MODEL_DIR

    def set_model(self, mtype, noise_level=None, eigv=None):
        self.mtype = mtype
        if mtype == 'DNN':
            self.cls = DNN()
        elif mtype == 'NzDNN':
            self.cls = NzDNN()
        else:
            raise ValueError('Unknown model type: {}'.format(mtype))

        self.cls.noise_level = noise_level
        if eigv is not None:
            self.cls.eigv = eigv

    def set_model_shape(self, input_dim, output_dim):
        self.cls.set_model_shape(input_dim, output_dim)
    
    def set_model_param(self, lr=0.01, dp=0.0, loss='mse', opt='adam'):
        self.cls.set_model_param(lr=lr, dp=dp, loss=loss, opt=opt)

    def set_tensorboard(self, name="", verbose=1):
        self.cls.save_dir = self.save_dir
        _=self.cls.set_tensorboard(self.log_dir, name, verbose)






    def prepare_DNN(self, input_dim, output_dim, lr=0.01, dp=0.0):
        self.cls.set_model_shape(input_dim, output_dim)
        self.cls.set_model_param(lr=lr, dp=dp, loss='mse', opt='adam', name='')
        self.cls.build_model()



    def add_noise(self, x, noise):
        return x + noise


    def build_model(self):
        self.cls.build_model()


