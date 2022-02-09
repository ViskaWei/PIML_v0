# import os
# import sys
# import h5py
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from PIML.nn.dnn.model.nzdnn import NzDNN
from PIML.util.util import Util


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

    def init(self, W, R, nFtr=10, odx=[0,1,2]):
        self.init_W(W)
        self.init_R(R)
        self.odx  = odx
        self.nOdx = len(odx)
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

    
    

    def init_from_BoxWR(self, BoxWR, odx=[0,1,2]):
        self.init_W(BoxWR.W)
        self.init_R(BoxWR.R)
        self.odx  = odx
        self.nOdx = len(odx)
        self.nFtr = BoxWR.topk
        self.eigv = BoxWR.eigv
        self.nn_scaler, self.nn_rescaler = self.setup_scaler(self.PhyMin, self.PhyMax, self.PhyRng, odx=self.odx)
        self.PhyLong =  [dnnWR.PhyLong[odx_i] for odx_i in self.odx]
        self.estimate_snr = BoxWR.estimate_snr
        self.get_random_pmt = lambda x: BoxWR.get_random_pmts(x, nPara=5, method="halton")
        self.nlList = BoxWR.Obs.nlList
        self.snrList = BoxWR.Obs.snrList
        def prepare_trainset(nTrain, pmts=None, noise_level=None, add_noise=True):
            x, y, p = BoxWR.prepare_trainset(nTrain, pmts=pmts, noise_level=noise_level, add_noise=add_noise, odx=self.odx, topk=self.nFtr)
            return x, y, p

        def prepare_testset(nTest, pmts=None, noise_level=1, seed=None):
            x, outputs = BoxWR.prepare_testset(nTest, pmts=pmts, noise_level=noise_level, topk=self.nFtr, seed=seed)
            p = outputs[:, self.odx]
            return x, p
        
        def prepare_noiseset(pmt, noise_level=1, nObs=1):
            x = BoxWR.prepare_noiseset(pmt, noise_level, nObs)
            return x

        return prepare_trainset, prepare_testset, prepare_noiseset

    # DNN model ---------------------------------------------------------------------------------
    def prepare_model(self, lr=0.01, dp=0.0, nEpoch=None):
        NN = BaseNN()
        NN.set_model(self.mtype, noise_level=self.train_NL, eigv=self.eigv)
        NN.set_model_shape(self.nFtr, self.nOdx)
        NN.set_model_param(lr=lr, dp=dp, loss='mse', opt='adam')
        if self.save: 
            nameR = self.R if nEpoch is None else self.R +"_ep"+str(nEpoch) +"_"
            name = nameR + self.name + "_"
            NN.set_tensorboard(name=name, verbose=1)
        NN.build_model()
        return NN.cls

    def init_train(self, mtype="NzDNN", name="", save=1, train_NL=None, nTrain=1000):
        self.save = save
        self.mtype = mtype
        self.name = name
        self.train_NL = train_NL or self.nlList[0]
        self.nTrain = nTrain


    def train(self, model=None, lr=0.01, dp=0.0, batch=16, nEpoch=100, verbose=1):
        if model is None: model = self.prepare_model(lr=lr, dp=dp, nEpoch=nEpoch)
        logging.info(model.name)
        add_noise = False if self.mtype[:2] == "Nz" else True
        self.x_train, self.y_train, self.p_train = self.prepare_trainset(self.nTrain, noise_level=self.train_NL, add_noise=add_noise)
        model.fit(self.x_train, self.y_train, nEpoch=nEpoch, batch=batch, verbose=verbose)
        model.nn_rescaler = lambda x: self.nn_rescaler(x)
        self.nn = model
        if self.save: model.save_model()


    def test(self, nTest=100, test_NL=None, pmts=None, seed=None):
        self.nTest = nTest        
        self.test_NL=test_NL or self.nlList[0]
        self.x_test, self.p_test = self.prepare_testset(nTest, pmts=pmts, noise_level=test_NL, seed=seed)
        self.p_pred = self.nn.scale_predict(self.x_test)

    
    def init_eval(self):
        self.PhyLong =  [BaseBox.PhyLong[odx_i] for odx_i in self.odx]
        self.init_plot()


            

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


    # def run(self, lr=0.01, dp=0.0, batch=16, nEpoch=100, verbose=1):
    #     self.dnn.set_model_param(lr=lr, dp=dp, loss='mse', opt='adam', name='')
    #     self.dnn.build_model()
    #     self.dnn.fit(self.x_train, self.y_train, nEpoch=nEpoch, batch=batch, verbose=verbose)
    #     self.y_pred = self.predict(self.x_test)

    # def predict(self, x_test):
    #     pred = self.dnn.predict(x_test)
    #     pred_params = self.nn_rescaler(pred)
    #     return pred_params

    def init_eval(self):
        self.init_plot()
        # snr = 
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




    