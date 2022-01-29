import os
import sys
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


    def init_train(self, out_idx=[0,1,2], mtype="DNN", train_NL=None, nTrain=1000, nTest=100):
        self.odx  = out_idx
        self.nOdx = len(out_idx)
        self.PhyLong =  [BoxW.PhyLong[odx_i] for odx_i in self.odx]
        self.setup_scalers(out_idx)
        self.mtype = mtype
        self.train_NL = train_NL
        self.nTrain = nTrain
        self.nTest = nTest


# DNN model ---------------------------------------------------------------------------------
    def prepare_model(self, lr=0.01, dp=0.0, batch=16):
        NN = BaseNN()
        NN.set_model(self.mtype, noise_level=self.train_NL, eigv=self.eigv)
        NN.set_model_shape(self.nFtr, self.nOdx)
        NN.set_model_param(lr=lr, dp=dp, loss='mse', opt='adam', name='')
        NN.build_model()
        return NN.cls

#train ---------------------------------------------------------------------------------

    def run(self, lr=0.01, dp=0.0, batch=16, nEpoch=100, verbose=1, model=None):
        for R0 in self.Rs:
            model_R0 = None if model is None else model[R0]
            self.train_R0(R0, model=model_R0, lr=lr, dp=dp, batch=batch, nEpoch=nEpoch, verbose=verbose)
            self.p_pred[R0] = self.test_R0(R0)
        self.init_eval()

    def test(self, noise_level=None, pmts=None, new=False):
        for R0 in self.Rs:
            self.p_pred[R0] = self.test_R0(R0, noise_level=noise_level, pmts=pmts, new=new) 
        self.eval()



    def train_R0(self, R0, model=None, lr=0.01, dp=0.0, batch=16, nEpoch=100, verbose=1):
        if model is None: model = self.prepare_model(lr=lr, dp=dp, batch=batch)
        add_noise = False if self.mtype[:5] == "Noise" else True
        x_train, y_train, _=self.prepare_trainset_R0(R0, self.nTrain, noise_level=self.train_NL, add_noise=add_noise, onPCA=self.onPCA, odx=self.odx)
        # model.predict
        model.nn_rescaler = lambda x: self.rescale(x, R0)
        model.fit(x_train, y_train, nEpoch=nEpoch, batch=batch, verbose=verbose)
        self.nn[R0] = model

    def test_R0(self, R0, noise_level=None, pmts=None, new=False):
        model = self.nn[R0]
        if noise_level is None: noise_level = self.train_NL
        self.test_NL = noise_level

        if self.x_test is None or self.x_test[R0] is None or new:
            self.x_test, self.p_test = self.prepare_testset(self.nTest, pmts=pmts, noise_level=noise_level, seed=922, odx=self.odx)
        p_pred = {}
        for R, x_test in self.x_test.items():
            p_pred[R] = model.predict(x_test, scaler=model.nn_rescaler)
        return p_pred

    

#eval ---------------------------------------------------------------------------------
    def init_eval(self):
        self.init_plot()
        # snr = self.estimate_snr(self.test_NL)
        self.eval_acc()
        self.eval_box()
        # pmts = self.get_random_pmt(10)
        # self.eval_pmts_noise(pmts, self.test_NL, nObs=100, n_box=0.2)

    def init_plot(self):
        self.PLT = PlotDNN(self.odx)
        self.PLT.pMaxs = self.pMaxs
        self.PLT.pMins = self.pMins
        self.PLT.pRngs = self.pRngs
        self.PLT.PhyLong = self.PhyLong
        # self.PLT.make_box_fn = lambda x: self.PLT.box_fn(self.pRng, self.pMin, 
        #                                                 self.pMax, n_box=x, 
        #                                                 c=BaseBox.DRC[self.R], RR=self.RR)

    def eval(self):
        self.eval_acc()
        self.eval_box()

    def eval_acc(self):
        for R0 in self.Rs:
            self.eval_acc_R0(R0)

    def eval_acc_R0(self, R0, snr=None, vertical=False):
        # if snr is None: snr = self.test_snr
        f, axs = self.PLT.plot_acc(self.p_pred[R0][R0], self.p_test[R0], self.pMins[R0], self.pMaxs[R0], 
                                    RR=BoxW.DRR[R0], c1=BoxW.DRC[R0], axes_name = self.PhyLong, vertical=vertical)
        if snr is None:
            f.suptitle(f"NL = {self.test_NL}")
        else:
            f.suptitle(f"SNR = {snr:.2f}")

    def eval_box(self, n_box=1, snr=None):
        for R0 in self.Rs:
            self.eval_box_R0(R0, n_box, snr)

    def eval_box_R0(self, R0, n_box=0.2, snr=None):
        fns = []
        for R1 in self.Rs:
            fns = fns + [self.PLT.scatter_fn(self.p_pred[R0][R1], c=BoxW.DRC[R1], s=1, lgd=None)]

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