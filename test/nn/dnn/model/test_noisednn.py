import numpy as np
from PIML.nn.dnn.model.nzdnn import NzDNN, DataGenerator 
from testbase import TestBase

class Test_NzDNN(TestBase):

    def test_init(self):
        self.dnn = NzDNN()
        self.N = 10
        self.dnn.set_model_shape(self.N,self.N)
        self.dnn.set_model_param()
        self.dnn.build_model(noise_level=1)

    def test_init_bnds(self):
        pass

    def test_DataGenerator(self):
        N = self.N
        x = np.array([np.arange(N), np.arange(N),np.arange(N),np.arange(N),np.arange(N), np.arange(N),np.arange(N),np.arange(N),np.arange(N),np.arange(N)])
        x[:,0] = np.arange(len(x))

# dnn.fit(x,x,x, batch=2, shuffle=False, ep=1)