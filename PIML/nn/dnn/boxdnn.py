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

from ...util.baseNN import BaseDNN


class BoxDNN(BaseDNN):
    def __init__(self, R0, top=100, pdx=[0,1,2], N_test=1000, step=20):
        super().__init__()
        self.pdx=pdx
        self.npdx=len(pdx)
        self.top=top
        self.N_test=N_test
        self.Wnms = ["RML"]
        self.Win = "RedM"
        self.R0= R0
        self.step=step
        # self.Wnms = ["BL","RML","NL"]
        self.n_ftr = top * len(self.Wnms)
        self.f_trains = {}
        self.f_tests = {}
        self.s_trains = {}
        self.s_tests = {}

# # data --------------------------------------------------------------------------------------------------

#     def prepare_R0(self, R0, R1=None, N_train=10000):
#         self.load_PCs(top=self.top)
#         self.setup_scalers()
#         self.prepare_trainset(R0, N_train)
#         if R1 is None:
#             self.prepare_testset_R0(R0, self.N_test)
#         else:
#             self.prepare_testset_R0_R1(R0, R1, self.N_test)

#     def load_PCs(self, top=None, W=None):
#         self.PCs[W[0]], self.nPixel = self.load_PC_W(Rs=[R0], W=W, top=top, step=20)



    def prepare_trainset_R0(self, R0, train_set, test_set):
        nsflux = self.add_noise(flux, err)
        # self.f_trains[R0] = nsflux
        self.x_trains[R0] = train_set
        self.y_trains[R0] = self.scale(pval, R0)
        self.p_trains[R0] = pval
        self.s_trains[R0] = snr



#     def prepare_testset_R0(self, R0, N_test):
#         for R1 in self.Rnms:
#             self.prepare_testset_R0_R1(R0, R1, N_test)

#     def prepare_testset_R0_R1(self, R0, R1, N_test):
#         wave, flux, err, pval, snr = self.load_nsRBF_data(R1, N_test, pdx=self.pdx)
#         nsflux = self.add_noise(flux, err)
#         self.f_tests[R1] = nsflux
#         self.x_tests[R1] = self.transform_R(nsflux, R0)
#         self.p_tests[R1] = pval
#         self.s_tests[R1] = snr

#     # def test_SN(self):
#     #     self.x_SNs = {}
#     #     self.y_SNs = {}
#     #     self.p_SNs = {}
#     #     self.s_SNs = {}

#     #     nsdx=0
#     #     SN = snr[nsdx][2]
#     #     nsfluxs = ddp.add_noise_N(flux[nsdx], err[nsdx], 1000)

#     def transform_R_W(self, x, R, W):
#         return x.dot(self.PCs[W][R].T)

# Train --------------------------------------------------------------------------------------------------

    def run_R0(self, R0, lr=0.01, dp=0.01, ep=1, verbose=0):
        dnn = self.prepare_DNN(lr=lr, dp=dp)
        dnn.fit(self.x_trains[R0], self.y_trains[R0], ep=ep, verbose=verbose)
        p_preds_R0= {}
        for R, x_test in self.x_tests.items():
            p_preds_R0[R] = self.predict(x_test, R0, dnn=dnn)
        self.p_preds[R0] = p_preds_R0
        self.dCT[R0] = self.get_overlap_R0(R0)
        self.dnns[R0] = dnn


    def predict_nsflux(self, nsflux, R0, dnn=None):
        x_test = self.transform_R(nsflux, R0)
        p_pred = self.predict(x_test, R0, dnn=dnn)
        return p_pred

    def eval_nsflux(self, SN):
        PATH = f"/scratch/ceph/swei20/data/dnn/BHB/snr{SN}_1k.h5"
        with h5py.File(PATH, "r") as f:
            nsflux = f["nsflux_R"][()]
            flux = f["flux_R"][()]