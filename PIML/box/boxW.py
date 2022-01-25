import os
import h5py
import numpy as np
from PIML import obs
from PIML.util.basebox import BaseBox
from PIML.util.baseplot import BasePlot

from PIML.obs.obs import Obs
from PIML.method.llh import LLH
from PIML.method.rbf import RBF
from PIML.method.bias import Bias
import matplotlib.pyplot as plt
from tqdm import tqdm

from PIML.util.util import Util

# class testBoxWR(BoxWR):


class BoxW(BaseBox):
    def __init__(self):
        super().__init__()
        self.W = None
        self.Rs = None
        self.RRs = None
        # self.LLH = {}
        self.Res = None
        self.topk = None
        self.onPCA = None

        self.DAk = {}
        self.DPara = {}
        self.DRbf_ak = {}
        self.DRbf_sigma = {}
        self.DRbf_flux = {}


    def init(self, W, Rs, Res, step, topk=10, onPCA=1):
        self.init_WR(W,Rs)
        self.Res = Res
        self.step = step
        self.topk = topk
        self.onPCA = onPCA
        for R in self.Rs:
            self.store_rbf(R)
        # self.init_plot_R()


    def store_rbf(self, R):
        print(f"=============================PREPARING {R}=====================")
        if R not in self.Rs:
            self.store_bnd(R)
            self.store_scaler(R)
        flux, pdx0, para = self.prepare_data_R(self.Res, R, self.step)
        interp_flux_fn, rbf_ak, rbf_sigma, interp_bias_fn = self.prepare_rbf(pdx0, self.pmt2pdx_scaler[R], 
                                                                            flux, onPCA=self.onPCA, Obs=self.Obs)
        if self.onPCA:
            error = abs(flux.dot(self.eigv) - rbf_ak(para)).sum()
        else:
            error = abs(np.log(flux) - interp_flux_fn(para, log=1, dotA=1)).sum()
        print(f"error: {error}")
        # self.DAk[R] = flux.dot(self.eigv.T)
        self.DPara[R] = para
        self.DRbf_ak[R] = rbf_ak                                                                
        self.DRbf_sigma[R] = rbf_sigma
        self.DRbf_flux[R] = interp_flux_fn
        return flux, interp_flux_fn, rbf_ak, 



    # def prepare_