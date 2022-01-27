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

        self.DPara = {}
        self.DV = {}
        self.DAk = {}
        self.DRbf_ak = {}
        self.DRbf_sigma = {}
        self.DRbf_flux = {}
        self.DRbf_noiz = {}


    def init(self, W, Rs, Res, step, topk=10, onPCA=1):
        self.init_WR(W,Rs)
        self.Res = Res
        self.step = step
        self.topk = topk
        self.onPCA = onPCA
        for R in self.Rs:
            self.run_step_rbf(R)
        self.set_eigv()
        # self.init_plot_R()

    def set_eigv(self):
        eigvs = []
        nFtr = 0
        for eigv in self.DV.values():
            eigvs.append(eigv)
            nFtr += len(eigv)
        self.eigv = np.vstack(eigvs)
        self.nFtr = nFtr

#rbf -----------------------------------------------------------------------------------------------------------------------
    def run_step_rbf(self, R):
        print(f"=============================PREPARING {R}=====================")
        if R not in self.Rs:
            self.store_bnd(R)
            self.store_scaler(R)
            self.Rs.append(R)
        flux, pdx0, para = self.prepare_data_R(self.Res, R, self.step)
        self.DPara[R] = para

        if self.onPCA:
            eigv, pcflux, fns = self.prepare_rbf(pdx0, self.pmt2pdx_scaler[R], 
                                                flux, onPCA=self.onPCA, Obs=self.Obs)
            rbf_flux, rbf_ak, rbf_sigma, gen_nObs_noise = fns
            error = abs(pcflux - rbf_ak(para)).sum()
            print(f"error: {error}")

            self.DV[R] = eigv
            self.DAk[R] = pcflux
            self.DRbf_ak[R] = rbf_ak                                                                
            self.DRbf_sigma[R] = rbf_sigma
            self.DRbf_flux[R] = rbf_flux
            self.DRbf_noiz[R] = gen_nObs_noise
        else:
            rbf_flux, rbf_sigma, gen_nObs_noise =  self.prepare_rbf(pdx0, self.pmt2pdx_scaler[R], 
                                                                            flux, onPCA=self.onPCA, Obs=self.Obs)
            error = abs(np.log(flux) - rbf_flux(para, log=1, dotA=1)).sum()
            print(f"error: {error}")
            self.DRbf_sigma[R] = rbf_sigma
            self.DRbf_flux[R] = rbf_flux


    def get_random_pmt_R(self, R, nPmt, nPara=5, method="halton"):
        rands = Util.get_random_uniform(nPmt, nPara, method=method, scaler=None)
        pmts = self.minmax_rescaler[R](rands)
        return rands, pmts

#DNN ----------------------------------------------------------------------------   
    def prepare_trainset(self, N, noise_level=1, add_noise=False):
        x_train, y_train, p_train = {}, {}, {}
        for R in self.Rs:
            pcfluxs, rands, pmts = self.prepare_trainset_R0(R, N, add_noise=add_noise, noise_level=noise_level)
            x_train[R] = pcfluxs
            y_train[R] = rands
            p_train[R] = pmts
        return x_train, y_train, p_train

    def prepare_trainset_R0(self, R0, N, pmts=None, noise_level=1, add_noise=False):
        if pmts is None: 
            rands, pmts = self.get_random_pmt_R(R0, N, method="random")
        else:
            pmts = pmts[:N]
            rands = self.minmax_scaler[R0](pmts)
        # aks = self.DRbf_ak[R](pmts)
        fluxs = self.DRbf_flux[R0](pmts)
        pcflux = fluxs.dot(self.eigv.T)
        assert pcflux.shape == (N, self.nFtr)
        print(f"generating {pcflux.shape} training data for {BaseBox.DRR[R0]}")

        if add_noise:
            if noise_level > 1:
                sigma = self.DRbf_sigma(pmts, noise_level, divide=0) 
                noiseMat = np.random.normal(0, sigma, sigma.shape)
                noisePC = noiseMat.dot(self.eigv.T)
                pcflux += noisePC
            out = pcflux
        else:
            if noise_level > 1:
                sigma = self.DRbf_sigma(pmts, 1, divide=0) # noise_level is 1 since noise will be added on the fly
                out = [pcflux, sigma]
            else:
                out = [pcflux, 0]
        return out, rands, pmts


    def prepare_testset_R0_R1(self, R0, R1, N, pmts=None, noise_level=1, seed=None):
        if pmts is None: 
            _, pmts = self.get_random_pmt_R(R1, N, method="random")
        else:
            pmts = pmts[:N]
        fluxs = self.DRbf_flux[R1](pmts)

        if noise_level > 1:            
            sigma = self.DRbf_sigma(pmts, noise_level, divide=0)
            if seed is not None: np.random.seed(seed)
            noiseMat = np.random.normal(0, sigma, sigma.shape)
            fluxs = fluxs + noiseMat

        pcflux = fluxs.dot(self.DV[R0].T) # convert noise into topk PC basis for R0
        return pcflux, pmts

    def prepare_testset_R0(self, R0, N, noise_level=1, seed=None):
        x_test, p_test={}, {}
        for R1 in self.Rs:
            x_test[R1], p_test[R1] = self.prepare_testset_R0_R1(R0, R1, N, noise_level=noise_level, seed=seed)
        return x_test, p_test

    def prepare_testset(self, N, noise_level, seed=None):
        x_test, p_test={}, {}
        for R0 in self.Rs:
            x_test[R0], p_test[R0] = self.prepare_testset_R0(R0, N, noise_level=noise_level, seed=seed)
        return x_test, p_test