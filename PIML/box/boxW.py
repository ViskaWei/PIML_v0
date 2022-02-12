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


    def init_box(self, W, Rs, Res, step, topk=10, onPCA=1, load_eigv=False, stack=False):
        self.init_WR(W,Rs)
        self.Res = Res
        self.step = step
        self.topk = topk
        self.onPCA = onPCA
        for R in self.Rs:
            self.run_step_rbf(R)
        if load_eigv: 
            self.DV = self.IO.load_eigv(topk=topk)
        if stack:
            self.eigv, self.nFtr = self.stack_eigv()
        else:
            self.eigv = self.DV
            self.nFtr = self.topk

    def save_eigv(self):
        self.IO.save_eigv(self.DV)

    def stack_eigv(self):
        nFtr = 0
        eigvs = []
        for eigv in self.DV.values():
            eigv0 = eigv[:self.topk]
            eigvs.append(eigv0)
            nFtr += len(eigv0)
        assert nFtr == self.topk*len(self.DV)
        return np.vstack(eigvs), nFtr

#rbf -----------------------------------------------------------------------------------------------------------------------
    def run_step_rbf(self, R, fast=False):
        logging.info(f"=============================PREPARING {R}=====================")
        if R not in self.Rs:
            self.store_bnd(R)
            self.store_scaler(R)
            self.Rs.append(R)
            self.RRs.append(BaseBox.DRR[R])
        flux, pdx0, para, self.Obs = self.prepare_data_WR(self.W,R,self.Res,self.step,store=False)
        self.DPara[R] = para

        if self.onPCA:
            eigv_R, aks, fns = self.prepare_PC_rbf(pdx0, self.pmt2pdx_scaler[R], 
                                                flux, Obs=self.Obs, topk=self.topk, fast=fast)
            rbf_flux, rbf_flux_sigma, rbf_ak, rbf_ak_sigma, rbf_logflux_sigma, rbf_logA, gen_nObs_noise = fns
            assert np.allclose(aks, rbf_ak(para))

            self.DV[R] = eigv_R
            self.DRbf_ak[R] = rbf_ak                                                                
            self.DRbf_flux_sigma[R] = rbf_logflux_sigma
            self.DRbf_flux[R] = rbf_flux
            # self.DRbf_noiz[R] = gen_nObs_noise
        else:
            rbf_flux, rbf_flux_sigma = self.prepare_logflux_rbf(pdx0, self.pmt2pdx_scaler[R], flux, Obs=self.Obs)
            error = abs(np.log(flux) - rbf_flux(para, log=1, dotA=1)).sum()
            logging.info(f"error: {error}")
            self.DRbf_flux_sigma[R] = rbf_flux_sigma
            self.DRbf_flux[R] = rbf_flux



#For TrainBoxW ----------------------------------------------------------------------------   
    def prepare_pmts(self, R0, N, pmts=None, odx=None):
        if pmts is None: 
            rands, pmts = self.get_random_pmts_R(R0, N, method="random", outRnd=True)
        else:
            pmts = pmts[:N]
            rands = self.minmax_scaler[R0](pmts)        
        if odx is not None: rands = rands[:,odx]
        return rands, pmts

    def prepare_trainset(self, N, noise_level=1, add_noise=False, pmts=None, odx=None):
        x_train, y_train, p_train = {}, {}, {}
        for R in self.Rs:
            pmts_R = None if pmts is None else pmts[R]
            aks, rands, pmts_R = self.prepare_trainset_R0(R, N, pmts=pmts_R, add_noise=add_noise, noise_level=noise_level, odx=None)
            x_train[R] = aks
            y_train[R] = rands
            p_train[R] = pmts_R
        return x_train, y_train, p_train

    def prepare_trainset_R0(self, R0, N, pmts=None, noise_level=1, add_noise=False, topk=None, eigv=None,  odx=None):
        rands, pmts = self.prepare_pmts(R0, N, pmts=pmts, odx=odx)
        if self.onPCA:
            out = self.prepare_ak_onPCA_R0(R0, pmts, eigv, topk=topk, noise_level=noise_level, add_noise=add_noise)
        else:
            #TODO: fix this
            pass
        if odx is not None: pmts = pmts[:, odx]
        return out, rands, pmts

    def prepare_ak_onPCA_R0(self, R0, pmts, eigv, noise_level=1, topk=None, add_noise=False):
        logfluxs, sigma_log = self.DRbf_flux_sigma[R0](pmts)
        if add_noise:
            obssigma = sigma_log * noise_level
            noise = np.random.normal(0, obssigma, obssigma.shape)
            logfluxs += noise
        aks = logfluxs.dot(eigv.T)
        logging.info(f"generating {aks.shape} training data for {BaseBox.DRR[R0]}")
        return aks if add_noise else [aks, sigma_log]
                
    def prepare_testset_R1(self, R1, N, pmts=None, noise_level=1, eigv=None, odx=None):
        pmts = self.get_random_pmts_R(R1, N, method="random", outRnd=False) if pmts is None else pmts[:N]
        if self.onPCA:
            out = self.prepare_logflux_on_PCA_R0(R1, pmts, noise_level=noise_level)
            if eigv is not None: out = out.dot(eigv.T)
        else:
            pass
        if odx is not None: pmts = pmts[:, odx]
        return out, pmts

    def prepare_testset(self, N, pmts=None, noise_level=1, eigv=None, odx=None):
        f_test, p_test={}, {}
        for R1 in self.Rs:
            pmts_R1 = None if pmts is None else pmts[R1]
            f_test[R1], p_test[R1] = self.prepare_testset_R1(R1, N, pmts=pmts_R1, noise_level=noise_level, eigv=eigv, odx=odx)
        return f_test, p_test

    def prepare_logflux_on_PCA_R0(self, R0, pmts, noise_level=1):
        logfluxs, sigma_log = self.DRbf_flux_sigma[R0](pmts)
        obssigma = sigma_log * noise_level
        noise = np.random.normal(0, obssigma, obssigma.shape)
        return logfluxs + noise
