import os
import sys
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.python.ops.gen_dataset_ops import prefetch_dataset
from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
tf.config.list_physical_devices('GPU') 
import warnings
warnings.filterwarnings("ignore")
import logging 
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from ...util.baseNN import BaseNN
from PIML.util.basebox import BaseBox
from .plotdnn import plotDNN

class dnnWR(BaseBox):
    def __init__(self):
        super().__init__()
        self.Wnms = ["RML"]
        self.Win = "RedM"

    def init(self, W, R, nFtr=10, out_idx=[0,1,2]):
        self.init_WR(W,R)
        self.odx  = out_idx
        self.nOdx = len(out_idx)
        self.nFtr = nFtr
        self.nn_scaler, self.nn_rescaler = self.setup_scaler(self.PhyMin, self.PhyMax, self.PhyRng, odx=self.odx)
        self.PhyLong =  [dnnWR.PhyLong[odx_i] for odx_i in self.odx]

    
    def init_dnn(self, lr=0.01, dp=0.0):
        NN = BaseNN()
        NN.set_model(type="DNN")
        NN.set_model_shape(self.nFtr, self.nOdx)
        NN.set_model_param(lr=lr, dp=dp, loss='mse', opt='adam', name='')
        NN.model.build_model()
        self.dnn = NN.model

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


    def prepro_input(self, data):
        input_data = data
        return input_data

    def prepro_output(self, data):
        output_data = self.nn_scaler(data)
        return output_data

    def prepare_trainset(self, inputs, outputs):
        x = self.prepro_input(inputs)
        p = outputs[:, self.odx]
        y = self.prepro_output(p)
        self.x_train, self.y_train, self.p_train = x, y, p
        
    def prepare_testset(self, inputs, outputs):
        x = self.prepro_input(inputs)
        p = outputs[:, self.odx]
        self.x_test, self.p_test = x, p

    def prepare(self, train_in, train_out, test_in, test_out):
        self.prepare_trainset(train_in, train_out)
        self.prepare_testset(test_in, test_out)

    def run(self, lr=0.01, dp=0.0, nEpoch=100, verbose=1):
        self.init_dnn(lr=lr, dp=dp)
        self.dnn.fit(self.x_train, self.y_train, ep=nEpoch, verbose=verbose)
        self.y_pred = self.predict(self.x_test)

    def predict(self, x_test):
        pred = self.dnn.predict(x_test)
        pred_params = self.nn_rescaler(pred)
        return pred_params

    def init_eval(self, noise_level, nl2snr_fn, gen_input):
        self.estimate_snr = nl2snr_fn
        self.gen_input = gen_input
        self.init_plot()
        snr = self.estimate_snr(noise_level)
        self.eval_acc(snr=snr)

    def init_plot(self):
        self.PLT = plotDNN()
        self.PLT.make_box_fn = lambda x: self.PLT.box_fn(self.pRng, self.pMin, 
                                                        self.pMax, n_box=x, 
                                                        c=BaseBox.DRC[self.R], RR=self.RR)


    def eval_acc(self, snr=1):
        f, axs = self.PLT.plot_acc(self.y_pred, self.p_test, self.pMin, self.pMax, RR=self.RR, axes_name = self.PhyLong)
        f.suptitle(f"SNR = {snr:.2f}")


    def eval_pmts_noise(self, pmts, noise_level, N_obs=10, n_box=0.5):
        fns = []
        snr = self.estimate_snr(noise_level)

        for pmt in tqdm(pmts):
            tests_pmt = self.gen_input(pmt, noise_level, N_obs=N_obs)
            preds_pmt = self.predict(tests_pmt)
            fns_pmt = self.PLT.flow_fn_i(preds_pmt, pmt[self.odx], legend=0)
            fns = fns + fns_pmt

        f = self.PLT.plot_box(self.odx, fns = fns, n_box=n_box)
        f.suptitle(f"SNR={snr:.2f}")

    def eval_pmt_noise(self, pmt, noise_level, N_obs, n_box=0.5):
        tests_pmt = self.gen_input(pmt, noise_level, N_obs=N_obs)
        preds = self.predict(tests_pmt)
        snr = self.estimate_snr(noise_level)
        self.PLT.plot_pmt_noise(preds, pmt[self.odx], self.odx, snr, n_box=n_box)
        return preds

    def estimate_snr(self, noise_level):
        pass

    def gen_input(self, pmt, noise_level, N_obs=1):
        pass

    