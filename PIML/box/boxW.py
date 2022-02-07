import logging
import numpy as np
from PIML.util.basebox import BaseBox
from PIML.util.util import Util

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
        self.DRbf_ak_sigma = {}
        self.DRbf_flux_sigma = {}


        self.DRbf_flux = {}
        self.DRbf_noiz = {}


    def init_box(self, W, Rs, Res, step, topk=10, onPCA=1, load_eigv=False):
        self.init_WR(W,Rs)
        self.Res = Res
        self.step = step
        self.topk = topk
        self.onPCA = onPCA
        for R in self.Rs:
            self.run_step_rbf(R)
        if load_eigv: 
            self.DV = self.load_eigv(topk=topk)
        self.set_eigv()
        # self.init_plot_R()

    def save_eigv(self):
        
        self.IO.save_eigv(self.DV)

    def load_eigv(self, topk=None, stack=0):
        DV = self.IO.load_eigv(topk=topk)
        if stack:
            self.DV = DV
            self.set_eigv()
        else:
            return DV

    def stack_eigv(self):
        nFtr = 0
        eigvs = []
        for eigv in self.DV.values():
            eigvs.append(eigv[:self.topk])
            nFtr += len(eigv)
        assert nFtr == self.topk*len(self.DV)
        return np.vstack(eigvs), nFtr

    def set_eigv(self):
        self.eigv, self.nFtr = self.stack_eigv()

        
        

#rbf -----------------------------------------------------------------------------------------------------------------------
    def run_step_rbf(self, R, fast=False):
        logging.info(f"=============================PREPARING {R}=====================")
        if R not in self.Rs:
            self.store_bnd(R)
            self.store_scaler(R)
            self.Rs.append(R)
            self.RRs.append(BaseBox.DRR[R])
        flux, pdx0, para = self.prepare_data_R(self.Res, R, self.step)
        self.DPara[R] = para

        if self.onPCA:
            eigv_R, pcflux, fns = self.prepare_rbf(pdx0, self.pmt2pdx_scaler[R], 
                                                flux, onPCA=self.onPCA, Obs=self.Obs, topk=self.topk, fast=fast)
            rbf_flux, rbf_ak, _, _, rbf_flux_sigma, gen_nObs_noise = fns
            error = abs(pcflux - rbf_ak(para)).sum()
            logging.info(f"error: {error}")

            self.DV[R] = eigv_R
            # self.DAk[R] = pcflux
            self.DRbf_ak[R] = rbf_ak                                                                
            self.DRbf_flux_sigma[R] = rbf_flux_sigma
            self.DRbf_flux[R] = rbf_flux
            # self.DRbf_noiz[R] = gen_nObs_noise
        else:
            rbf_flux, rbf_sigma, gen_nObs_noise =  self.prepare_rbf(pdx0, self.pmt2pdx_scaler[R], 
                                                                            flux, onPCA=self.onPCA, Obs=self.Obs)
            error = abs(np.log(flux) - rbf_flux(para, log=1, dotA=1)).sum()
            logging.info(f"error: {error}")
            self.DRbf_sigma[R] = rbf_sigma
            self.DRbf_flux[R] = rbf_flux


    def get_random_pmt_R(self, R, nPmt, nPara=5, method="halton"):
        rands = Util.get_random_uniform(nPmt, nPara, method=method, scaler=None)
        pmts = self.minmax_rescaler[R](rands)
        return rands, pmts

#For TrainBoxW ----------------------------------------------------------------------------   
    def prepare_pmts(self, R0, N, pmts=None, odx=None):
        if pmts is None: 
            rands, pmts = self.get_random_pmt_R(R0, N, method="random")
        else:
            pmts = pmts[:N]
            rands = self.minmax_scaler[R0](pmts)        
        if odx is not None: rands = rands[:,odx]
        return rands, pmts

    def prepare_trainset(self, N, noise_level=1, add_noise=False, pmts=None, odx=None):
        x_train, y_train, p_train = {}, {}, {}
        for R in self.Rs:
            pmts_R = None if pmts is None else pmts[R]
            pcfluxs, rands, pmts_R = self.prepare_trainset_R0(R, N, pmts=pmts_R, add_noise=add_noise, noise_level=noise_level, onPCA=self.onPCA, odx=None)
            x_train[R] = pcfluxs
            y_train[R] = rands
            p_train[R] = pmts_R
        return x_train, y_train, p_train

    def prepare_trainset_R0(self, R0, N, pmts=None, noise_level=1, add_noise=False, onPCA=1, odx=None):
        rands, pmts = self.prepare_pmts(R0, N, pmts=pmts, odx=odx)

        if noise_level <1:
            logfluxs = self.DRbf_flux[R0](pmts, log=1, dotA=0, outA=0) #dotA=0 or 1 is the same as its orthogonal to all box PCs.
            if onPCA:
                pcfluxs = logfluxs.dot(self.eigv.T)
                assert pcfluxs.shape == (N, self.nFtr)
                logging.info(f"generating {pcfluxs.shape} training data for {BaseBox.DRR[R0]}")
                out = pcfluxs if add_noise else [pcfluxs, np.zeros_like(pcfluxs)]
            else:
                out = logfluxs if add_noise else [logfluxs, np.zeros_like(logfluxs)]
        else:
            if add_noise:
                out = self.aug_flux_for_pmts_R(R0, pmts, noise_level)
            else:
                logModel, sigma_log  = self.DRbf_flux_sigma[R0](pmts)
                if self.onPCA: logModel = logModel.dot(self.eigv.T) 
                out = [logModel, sigma_log]
            if odx is not None: pmts = pmts[:, odx]
        return out, rands, pmts
            
    def prepare_testset_R1(self, R1, N, pmts=None, noise_level=1, seed=None, odx=None):
        pmts = self.get_random_pmt_R(R1, N, method="random")[1] if pmts is None else pmts[:N]

        if noise_level > 1:      
            logfluxs = self.aug_flux_for_pmts_R(R1, pmts, noise_level, seed=seed)
        else:
            logfluxs = self.DRbf_flux[R1](pmts, log=1, dotA=0, outA=0) #dotA=0 or 1 is the same as its orthogonal to all box PCs.
            if self.onPCA: logfluxs = logfluxs.dot(self.eigv.T)

        if odx is not None: pmts = pmts[:, odx]
        return logfluxs, pmts

    def prepare_testset(self, N, pmts=None, noise_level=1, seed=None, odx=None):
        x_test, p_test={}, {}
        for R1 in self.Rs:
            pmts_R1 = None if pmts is None else pmts[R1]
            x_test[R1], p_test[R1] = self.prepare_testset_R1(R1, N, pmts=pmts_R1, noise_level=noise_level, seed=seed, odx=odx)
        return x_test, p_test

    def aug_flux_for_pmts_R(self, R, pmts, noise_level, seed=None):
        logModel, sigma_log = self.DRbf_flux_sigma[R](pmts)
        if seed is not None: np.random.seed(seed)
        sigma = sigma_log * noise_level
        logNoise = np.random.normal(0, sigma, sigma.shape)
        logObsfluxs = logModel + logNoise
        return logObsfluxs.dot(self.eigv.T) if self.onPCA else logObsfluxs