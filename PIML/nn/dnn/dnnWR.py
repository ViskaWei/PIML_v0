from mimetypes import init
import os
import sys
from argon2 import PasswordHasher
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.python.ops.gen_dataset_ops import prefetch_dataset
from tqdm import tqdm

from PIML.nn.dnn.model.nzdnn import NzDNN
from PIML.util.util import Util


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
tf.config.list_physical_devices('GPU') 
import warnings
warnings.filterwarnings("ignore")
import logging 
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from ...util.baseNN import BaseNN
from PIML.util.basebox import BaseBox
from .plotdnn import PlotDNN

class dnnWR(BaseBox):
    def __init__(self, BoxWR = None):
        super().__init__()
        self.Wnms = ["RML"]
        self.Win = "RedM"
        if BoxWR is not None:
            self.prepare_trainset, self.prepare_testset, self.prepare_noiseset = self.init_from_BoxWR(BoxWR)

    def init(self, W, R, nFtr=10, out_idx=[0,1,2]):
        self.init_WR(W,R)
        self.odx  = out_idx
        self.nOdx = len(out_idx)
        self.nFtr = nFtr
        self.nn_scaler, self.nn_rescaler = self.setup_scaler(self.PhyMin, self.PhyMax, self.PhyRng, odx=self.odx)
        self.PhyLong =  [dnnWR.PhyLong[odx_i] for odx_i in self.odx]

        def prepro_input(data):
            input_data = data
            return input_data

        def prepro_output(data):
            output_data = self.nn_scaler(data)
            return output_data

        def prepare_trainset(inputs, outputs):
            x = self.prepro_input(inputs)
            p = outputs[:, self.odx]
            y = self.prepro_output(p)
            self.x_train, self.y_train, self.p_train = x, y, p
                
        def prepare_testset(inputs, outputs):
            x = self.prepro_input(inputs)
            p = outputs[:, self.odx]
            self.x_test, self.p_test = x, p

        def prepare(train_in, train_out, test_in, test_out):
            prepare_trainset(train_in, train_out)
            prepare_testset(test_in, test_out)
        return prepare

    

    def init_from_BoxWR(self, BoxWR, out_idx=[0,1,2]):
        self.init_WR(BoxWR.W, BoxWR.R)
        self.odx  = out_idx
        self.nOdx = len(out_idx)
        self.nFtr = BoxWR.topk
        self.eigv = BoxWR.eigv
        self.nn_scaler, self.nn_rescaler = self.setup_scaler(self.PhyMin, self.PhyMax, self.PhyRng, odx=self.odx)
        self.PhyLong =  [dnnWR.PhyLong[odx_i] for odx_i in self.odx]
        self.estimate_snr = BoxWR.estimate_snr
        self.get_random_pmt = lambda x: BoxWR.get_random_pmt(x, nPara=5, method="halton")

        def prepare_trainset(nTrain, pmts=None, noise_level=None, add_noise=True):
            x, outputs = BoxWR.prepare_trainset(nTrain, pmts=pmts, noise_level=noise_level, add_noise=add_noise)
            p = outputs[:, self.odx]
            y = self.nn_scaler(p)
            return x, y, p

        def prepare_testset(nTest, pmts=None, noise_level=1, seed=None):
            x, outputs = BoxWR.prepare_testset(nTest, pmts=pmts, noise_level=noise_level, seed=seed)
            p = outputs[:, self.odx]
            return x, p
        
        def prepare_noiseset(pmt, noise_level=1, nObs=1):
            x = BoxWR.prepare_noiseset(pmt, noise_level, nObs)
            return x

        return prepare_trainset, prepare_testset, prepare_noiseset

    def prepare_model(self, mtype="DNN", train_NL=None, tb=0, nTrain=1000, nTest=100):
        NN = BaseNN()
        self.train_NL = train_NL
        NN.set_model(mtype, noise_level=train_NL, eigv=self.eigv)
        NN.set_model_shape(self.nFtr, self.nOdx)
        if tb:
            self.log_path =NN.set_tensorboard(verbose=1)
        self.dnn = NN.cls

        
    def prepare_data(self, nTrain=1000, nTest=100, test_NL=1, seed=None):
        self.nTrain = nTrain
        self.nTest = nTest
        add_noise=True if self.dnn.mtype == "DNN" else False
        self.x_train, self.y_train, self.p_train = self.prepare_trainset(nTrain, noise_level=self.train_NL, add_noise=add_noise)
        
        self.test_NL=test_NL
        self.x_test, self.p_test = self.prepare_testset(nTest, noise_level=test_NL, seed=seed)
            

    def setup_scaler(self, PhyMin, PhyMax, PhyRng, odx=None):
        if odx is None: odx = self.odx
        self.pMin = PhyMin[self.odx]
        self.pRng = PhyRng[self.odx]
        self.pMax = PhyMax[self.odx]
        def scaler(x):
            return (x - self.pMin) / self.pRng
        def rescaler(x):
            return x * self.pRng + self.pMin
        return scaler, rescaler


    def run(self, lr=0.01, dp=0.0, batch=16, nEpoch=100, verbose=1):
        self.dnn.set_model_param(lr=lr, dp=dp, loss='mse', opt='adam', name='')
        self.dnn.build_model()
        self.dnn.fit(self.x_train, self.y_train, nEpoch=nEpoch, batch=batch, verbose=verbose)
        self.y_pred = self.predict(self.x_test)

    def predict(self, x_test):
        pred = self.dnn.predict(x_test)
        pred_params = self.nn_rescaler(pred)
        return pred_params

    def init_eval(self):
        self.init_plot()
        snr = self.estimate_snr(self.test_NL)
        self.eval_acc(snr)
        pmts = self.get_random_pmt(10)
        self.eval_pmts_noise(pmts, self.test_NL, nObs=100, n_box=0.2)

    def init_plot(self):
        self.PLT = PlotDNN()
        self.PLT.make_box_fn = lambda x: self.PLT.box_fn(self.pRng, self.pMin, 
                                                        self.pMax, n_box=x, 
                                                        c=BaseBox.DRC[self.R], RR=self.RR)


    def eval_acc(self, snr=None):
        if snr is None: snr = self.test_snr
        f, axs = self.PLT.plot_acc(self.y_pred, self.p_test, self.pMin, self.pMax, RR=self.RR, axes_name = self.PhyLong)
        f.suptitle(f"SNR = {snr:.2f}")


    def eval_pmts_noise(self, pmts, noise_level, nObs=10, n_box=0.5):
        fns = []
        snr = self.estimate_snr(noise_level)

        for pmt in tqdm(pmts):
            tests_pmt = self.prepare_noiseset(pmt, noise_level, nObs)
            preds_pmt = self.predict(tests_pmt)
            fns_pmt = self.PLT.flow_fn_i(preds_pmt, pmt[self.odx], legend=0)
            fns = fns + fns_pmt

        f = self.PLT.plot_box(self.odx, fns = fns, n_box=n_box)
        f.suptitle(f"SNR={snr:.2f}")

    def eval_pmt_noise(self, pmt, noise_level, nObs, n_box=0.5):
        tests_pmt = self.gen_input(pmt, noise_level, nObs=nObs)
        preds = self.predict(tests_pmt)
        snr = self.estimate_snr(noise_level)
        self.PLT.plot_pmt_noise(preds, pmt[self.odx], self.odx, snr, n_box=n_box)
        return preds




    