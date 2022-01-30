import os
import sys
import h5py
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import tensorflow as tf
tf.config.list_physical_devices('GPU') 
import warnings
warnings.filterwarnings("ignore")
import logging 
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from ...util.baseNN import BaseNN
from PIML.box.boxW import BoxW
from PIML.nn.dnn.plotdnn import PlotDNN


class DnnBoxW(BoxW):
    def __init__(self):
        super().__init__()
        self.pRngs = {}
        self.pMins = {}
        self.pMaxs = {}
        self.nn = {}
        self.x_test = None
        self.test_NL = None
        self.p_pred = {}


    def init_train(self, out_idx=[0,1,2], mtype="DNN", log=1, train_NL=None, nTrain=1000):
        self.odx  = out_idx
        self.nOdx = len(out_idx)
        self.PhyLong =  [BoxW.PhyLong[odx_i] for odx_i in self.odx]
        self.setup_scalers(out_idx)
        self.mtype = mtype
        self.train_NL = train_NL
        self.nTrain = nTrain
        self.log=log
        self.init_eval()

    def load_train(self, model_names, out_idx=[0,1,2], topk=10):
        self.topk = topk
        self.odx  = out_idx
        self.nOdx = len(out_idx)
        self.PhyLong =  [BoxW.PhyLong[odx_i] for odx_i in self.odx]
        self.setup_scalers(out_idx)
        self.load_eigv(self.topk, stack=1)
        for model_name in model_names:
            self.set_model_R0(model_name)
        self.init_eval()
        
    def load_model(self, model_name, path=None):
        if path is None:
            path = os.path.join(BoxW.MODEL_DIR, model_name, "model.h5")
        model = tf.keras.models.load_model(path)
        return model

    def set_model_R0(self, model_name):
        R0 = model_name[:1]
        model = self.load_model(model_name)
        def _predict(x):
            y = model.predict(x)
            return self.rescale(y, R0)            
        model._predict = _predict
        self.nn[R0] = model

# DNN model ---------------------------------------------------------------------------------
    def prepare_model_R0(self, R0, lr=0.01, dp=0.0, batch=16, nEpoch=None):
        NN = BaseNN()
        NN.set_model(self.mtype, noise_level=self.train_NL, eigv=self.eigv)
        NN.set_model_shape(self.nFtr, self.nOdx)
        NN.set_model_param(lr=lr, dp=dp, loss='mse', opt='adam')
        if self.log: 
            name = R0 if nEpoch is None else R0+"_ep"+str(nEpoch) +"_"
            NN.set_tensorboard(name=name, verbose=1)
        NN.build_model()
        return NN.cls

#train ---------------------------------------------------------------------------------

    def run(self, lr=0.01, dp=0.0, batch=16, nEpoch=100, verbose=1, model=None, eval=1):
        for R0 in self.Rs:
            model_R0 = None if model is None else model[R0]
            self.train_R0(R0, model=model_R0, lr=lr, dp=dp, batch=batch, nEpoch=nEpoch, verbose=verbose)
        
        if eval:
            self.test(test_NL=self.train_NL, nTest=100)
            



    def train_R0(self, R0, model=None, lr=0.01, dp=0.0, batch=16, nEpoch=100, verbose=1):
        if model is None: model = self.prepare_model_R0(R0, lr=lr, dp=dp, batch=batch, nEpoch=nEpoch)
        logging.info(model.name)
        model.R0 = R0
        add_noise = False if self.mtype[:2] == "Nz" else True
        x_train, y_train, _=self.prepare_trainset_R0(R0, self.nTrain, noise_level=self.train_NL, add_noise=add_noise, onPCA=self.onPCA, odx=self.odx)
        # model.predict
        model.nn_rescaler = lambda x: self.rescale(x, R0)
        model.fit(x_train, y_train, nEpoch=nEpoch, batch=batch, verbose=verbose)
        self.nn[R0] = model
        if self.log: model.save_model()
            

    def test(self, test_NL=None, nTest=100, pmts=None, new=False):
        self.nTest = nTest
        self.test_NL = test_NL
        self.nnRs = list(self.nn.keys())
        if self.x_test is None or new:
            self.x_test, self.p_test = self.prepare_testset(self.nTest, pmts=pmts, noise_level=test_NL, seed=922, odx=self.odx)
        
        for R0 in self.nnRs:
            self.p_pred[R0] = self.test_R0(R0, self.x_test) 
        vertical = 1 if self.nR > 2 else 0
        self.eval(self.p_pred, self.p_test, vertical=vertical)

    def test_R0(self, R0, x_test):
        model = self.nn[R0]
        p_pred = {}
        for R, x_test in self.x_test.items():
            p_pred[R] = model._predict(x_test)
        return p_pred

    

#eval ---------------------------------------------------------------------------------
    def init_eval(self):
        self.init_plot()
        # snr = self.estimate_snr(self.test_NL)
        

        # pmts = self.get_random_pmt(10)
        # self.eval_pmts_noise(pmts, self.test_NL, nObs=100, n_box=0.2)

    def init_plot(self):
        self.PLT = PlotDNN(self.odx)
        self.PLT.Rs = self.Rs
        self.PLT.RRs = self.RRs
        self.PLT.pMaxs = self.pMaxs
        self.PLT.pMins = self.pMins
        self.PLT.pRngs = self.pRngs
        self.PLT.PhyLong = self.PhyLong
        # self.PLT.make_box_fn = lambda x: self.PLT.box_fn(self.pRng, self.pMin, 
        #                                                 self.pMax, n_box=x, 
        #                                                 c=BaseBox.DRC[self.R], RR=self.RR)

    def eval(self, p_pred, p_test, vertical=0):
        self.eval_acc(p_pred, p_test, vertical=vertical)
        self.eval_box(p_pred)

    def eval_acc(self, p_pred, p_test, vertical=False):
        for R0 in p_pred.keys():
            self.eval_acc_R0(R0, p_pred, p_test, vertical=vertical)

    def eval_acc_R0(self, R0, p_pred, p_test, snr=None, vertical=False):
        f, axs = self.PLT.plot_acc(p_pred[R0][R0], p_test[R0], self.pMins[R0], self.pMaxs[R0], 
                                    RR=BoxW.DRR[R0], c1=BoxW.DRC[R0], axes_name = self.PhyLong, vertical=vertical)
        if snr is None:
            f.suptitle(f"NL = {self.test_NL}")
        else:
            f.suptitle(f"SNR = {snr:.2f}")

    def eval_box(self, p_pred, n_box=1, snr=None):
        crossMat = self.PLT.get_crossMat(p_pred)
        for R0 in self.nnRs:
            self.eval_box_R0(R0, crossMat=crossMat, n_box=n_box, snr=snr)

    def eval_box_R0(self, R0, crossMat=None, n_box=0.2, snr=None):
        fns = []
        if crossMat is not None:
            R0_idx = self.nnRs.index(R0)
            crossMat_R0 = crossMat[R0_idx]
        for ii, R1 in enumerate(self.Rs):
            lgd = None if crossMat is None else crossMat_R0[ii]   
            fns = fns + [self.PLT.scatter_fn(self.p_pred[R0][R1], c=BoxW.DRC[R1], s=1, lgd=lgd)]

        f = self.PLT.plot_box_R0(R0, fns = fns, n_box=n_box)
        if snr is None:
            f.suptitle(f"NL = {self.test_NL}")
        else:
            f.suptitle(f"SNR = {snr:.2f}")




# scaler ---------------------------------------------------------------------------------
    def setup_scalers(self, odx=None):
        if odx is None: odx = self.odx
        for R in self.Rs:
            self.pRngs[R], self.pMins[R], self.pMaxs[R] = self.get_scaler(R, odx)


    def get_scaler(self, R, odx):
        pRng = self.DPhyRng[R][odx]
        pMin = self.DPhyMin[R][odx]
        pMax = self.DPhyMax[R][odx]
        return pRng, pMin, pMax

    def scale(self, pval, R):
        pnorm = (pval - self.pMins[R]) / self.pRngs[R]        
        return pnorm

    def rescale(self, pnorm, R):
        pval = pnorm * self.pRngs[R] + self.pMins[R]
        return pval