import os
import logging

from ..util.baseNN import BaseNN
from PIML.box.boxW import BoxW


class TrainBoxW(BoxW):
    def __init__(self):
        super().__init__()
        self.nn = {}
        self.name= None


    def init_train(self, odx=[0,1,2], mtype="DNN", save=1, train_NL=None, nTrain=1000, name=""):
        self.odx  = odx
        self.nOdx = len(odx)
        self.mtype = mtype
        self.train_NL = train_NL
        self.nTrain = nTrain
        self.save=save
        self.name = name

        

# DNN model ---------------------------------------------------------------------------------
    def prepare_model_R0(self, R0, lr=0.01, dp=0.0, nEpoch=None):
        NN = BaseNN()
        eigvk = self.DV[R0][:self.topk] if isinstance(self.eigv, dict) else self.eigv 
        NN.set_model(self.mtype, noise_level=self.train_NL, eigv=eigvk)
        NN.set_model_shape(self.nFtr, self.nOdx)
        NN.set_model_param(lr=lr, dp=dp, loss='mse', opt='adam')
        if self.save: 
            nameR = R0 if nEpoch is None else R0+"_ep"+str(nEpoch) +"_"
            name = nameR + self.name + "_"
            NN.set_tensorboard(name=name, verbose=1)
        NN.build_model()
        return NN.cls

#train ---------------------------------------------------------------------------------

    def run(self, lr=0.01, dp=0.0, batch=16, nEpoch=100, verbose=1, model=None):
        for R0 in self.Rs:
            model_R0 = None if model is None else model[R0]
            self.train_R0(R0, model=model_R0, lr=lr, dp=dp, batch=batch, nEpoch=nEpoch, verbose=verbose)
            
    def train_R0(self, R0, model=None, lr=0.01, dp=0.0, batch=16, nEpoch=100, verbose=1):
        if model is None: model = self.prepare_model_R0(R0, lr=lr, dp=dp, nEpoch=nEpoch)
        logging.info(model.name)
        model.R0 = R0
        add_noise = False if self.mtype[:2] == "Nz" else True
        x_train, y_train, _=self.prepare_trainset_R0(R0, self.nTrain, noise_level=self.train_NL, 
                                            eigv=model.eigv, add_noise=add_noise, odx=self.odx)
        logging.info(f"{x_train[0].shape}, {x_train[1].shape}, {y_train.shape}")
        model.fit(x_train, y_train, nEpoch=nEpoch, batch=batch, verbose=verbose)
        self.nn[R0] = model
        if self.save: model.save_model()
            




    



