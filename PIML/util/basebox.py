import numpy as np
import pandas as pd
import logging
from .util import Util
from .baseplot import BasePlot
from .basespec import BaseSpec
from PIML.obs.obs import Obs
from PIML.obs.obsW import ObsW

from PIML.method.rbf import RBF


class BaseBox(Util):


    def __init__(self):
        super().__init__()
        self.PLT = BasePlot()
        self.onPCA = None


# init ------------------------------------------------------------------------
    def init_W(self, W):
        self.W = W
        self.WRng = BaseBox.DWs[W]
        self.wave = None

    def init_Ws(self, Ws):
        if isinstance(Ws, str): Ws = [Ws]
        self.Ws = Ws
        self.DWs = {W: BaseBox.DWs[W] for W in Ws}
        self.DWv = {}


    def get_flux_in_Wrng_W(self, W, wave_H, flux_H):
        return BaseSpec._get_flux_in_Wrng(wave_H, flux_H, BaseBox.DWs[W])

    def get_flux_in_Wrng(self, wave, flux):
        return BaseSpec._get_flux_in_Wrng(wave, flux, self.WRng)

    def init_WR(self, W, Rs):
        self.init_W(W)
        self.init_Rs(Rs)



    def init_R(self, R):
        self.R = R
        self.RR = BaseBox.DRR[R]
        self.PhyMin, self.PhyMax, self.PhyRng, self.PhyNum, self.PhyMid = self.get_bnd(R)
        self.minmax_scaler, self.minmax_rescaler = BaseBox.get_minmax_scaler_fns(self.PhyMin, self.PhyRng)
        self.pmt2pdx_scaler, _ = BaseBox.get_pdx_scaler_fns(self.PhyMin)

    def init_Rs(self, Rs):
        if isinstance(Rs, str): Rs = [Rs]
        self.Rs = Rs
        self.RRs = [BaseBox.DRR[R] for R in Rs]
        self.nR = len(Rs)
        self.init_bnds(Rs)
        self.init_scalers()

    def init_bnds(self, Rs=None):
        self.DPhyMin = {}
        self.DPhyMax = {}
        self.DPhyRng = {}
        self.DPhyNum = {}
        self.DPhyMid = {}

        if Rs is None: Rs = BaseBox.Rnms
        if isinstance(Rs, str): Rs = [Rs]

        for R in Rs:
            self.store_bnd(R)

    def store_bnd(self, R):
        PhyMin, PhyMax, PhyRng, PhyNum, PhyMid = self.get_bnd(R)
        self.DPhyMin[R] = PhyMin
        self.DPhyMax[R] = PhyMax
        self.DPhyRng[R] = PhyRng
        self.DPhyNum[R] = PhyNum
        self.DPhyMid[R] = PhyMid        

    def init_scalers(self):
        self.minmax_scaler = {}
        self.minmax_rescaler = {}
        self.pmt2pdx_scaler = {}

        for R in self.DPhyMin.keys():
            self.store_scaler(R)

    def store_scaler(self, R):
        PhyMin = self.DPhyMin[R]
        self.minmax_scaler[R], self.minmax_rescaler[R] = BaseBox.get_minmax_scaler_fns(PhyMin, self.DPhyRng[R])
        self.pmt2pdx_scaler[R], _ = BaseBox.get_pdx_scaler_fns(PhyMin)

    def init_plot_R(self):
        self.PLT.make_box_fn = lambda x: self.PLT.box_fn(self.PhyRng, self.PhyMin, 
                                                        self.PhyMax, n_box=x, 
                                                        c=BaseBox.DRC[self.R], RR=self.RR)

# dataloader ------------------------------------------------------------------
    def prepare_data_R(self, W, R, Res, step, store=False):
        wave_H, flux_H, pdx, para = self.IO.load_bosz(Res, RR=BaseBox.DRR[R])
        pdx0 = pdx - pdx[0]
        if store:
            self.Mdx = Util.get_fdx_from_pmt(self.PhyMid, para)
            self.DWv_H = {}
            self.DFx_H = {}
            self.DFx_H0 = {}
            self.DFx0 = {}

        for W in self.Ws:
            wave_H, flux_H = self.get_flux_in_Wrng_W(W, wave_H, flux_H)
            wave, flux = BaseSpec.resample(wave_H, flux_H, step)
            self.DWv[W] = wave
            if store:
                self.DWv_H[W] = wave_H
                self.DFx_H[W] = flux_H
                self.DFx_H0[W] = flux_H[self.Mdx]
                self.DFx0[W] = flux[self.Mdx]
                self.Obs = self.init_obs(wave_H, step, flux_in_res=self.flux_H0)
            else:
                self.init_obs(wave_H, step, flux_in_res=None)

    def prepare_data_WR(self, W, R, Res, step, store=False):
        wave_H, flux_H, pdx, para = self.IO.load_bosz(Res, RR=BaseBox.DRR[R])
        pdx0 = pdx - pdx[0]
        wave_H, flux_H = self.get_flux_in_Wrng_W(W, wave_H, flux_H)
        wave, flux = BaseSpec.resample(wave_H, flux_H, step)
        if self.wave is None: 
            self.wave = wave
        else:
            assert np.all(self.wave == wave)
        if store:
            self.wave_H = wave_H
            self.Mdx = Util.get_fdx_from_pmt(self.PhyMid, para)
            self.flux_H = flux_H
            self.flux_H0 = flux_H[self.Mdx]
            self.flux0 = flux[self.Mdx]
            
        flux_in_res = self.flux_H0 if store else None
        Obs = self.init_obs_W(W, wave_H, step, flux_in_res=flux_in_res)
        return flux, pdx0, para, Obs
        
#rbf ---------------------------------------------------------------------------
    def prepare_logflux_rbf(self, coord, coord_scaler, flux, Obs=None):
        logflux = Util.safe_log(flux)
        rbf = RBF(coord=coord, coord_scaler=coord_scaler)
        rbf_flux = rbf.build_logflux_rbf_interp(logflux)
        if Obs is None:
            return rbf_flux, None
        else:
            def rbf_flux_sigma(pmt):
                Model = rbf_flux(pmt, log=0)
                sigma = Obs.get_sigma_from_flux(flux, noise_level=1)
                return Model, sigma
            return rbf_flux, rbf_flux_sigma

    def prepare_PC_rbf(self, coord, coord_scaler, flux, Obs=None, topk=None, fast=False):
        logflux = Util.safe_log(flux)
        rbf = RBF(coord=coord, coord_scaler=coord_scaler)
        top = topk if fast else None
        logA, aks, eigv = self.prepare_pca(logflux, top=top)
        rbf_flux, rbf_logA, rbf_ak = rbf.build_PC_rbf_interp(logA, aks, eigv)
        if Obs is None: return eigv, aks, [rbf_flux, rbf_ak, None, None] 

        def rbf_flux_sigma(pmt):
            AModel = rbf_flux(pmt, log=0, dotA=1, outA=0)
            sigma = Obs.get_sigma_from_flux(AModel, 1) # noise level = 1
            return [AModel, sigma]

        def rbf_ak_sigma(pmt, topk=None):
            ak, AModel = rbf_flux(pmt, log=0, dotA=1, outA=1)
            sigma = Obs.get_sigma_from_flux(AModel, 1) # noise level = 1
            sigma_log = np.divide(sigma, AModel)
            return [ak[...,:topk], sigma_log]

        def rbf_logflux_sigma(pmt):
            logAModel = rbf_flux(pmt, log=1, dotA=1, outA=0)
            AModel = np.exp(logAModel)
            sigma = Obs.get_sigma_from_flux(AModel, 1) # noise level = 1
            sigma_log = sigma / AModel
            return [logAModel, sigma_log]

        def gen_nObs_noise(sigma, nObs=1, topk=None):
            if nObs > 1: sigma = np.tile(sigma, (nObs, 1))
            noise = np.random.normal(0, sigma, sigma.shape)
            return noise.dot(eigv[:topk].T)
        
        return eigv, aks, [rbf_flux, rbf_flux_sigma, rbf_ak, rbf_ak_sigma, rbf_logflux_sigma, rbf_logA, gen_nObs_noise]




# PCA --------------------------------------------------------------------------
    def prepare_pca(self, logfluxs, top=None):
        logA = np.mean(logfluxs, axis=1)
        lognormModels = logfluxs - logA[:, None]
        u,s,v = np.linalg.svd(lognormModels, full_matrices=False)
        if top is not None: 
            u = u[:, :top]
            s = s[:top]
            v = v[:top]
            self.pcaDim = len(v)
        nS = len(s)
        logging.info(f"Top #{nS} eigs {s[:10].round(2)}")
        assert np.allclose(v.dot(v.T), np.eye(nS))
        assert np.allclose(np.sum(v[:-1], axis=1).mean(), 0.0) 

        aks = u * s
        assert np.allclose(lognormModels.dot(v.T), aks)
        return logA, aks, v
    
    
    def prepare_pca(self, logfluxs, top=None):
        nPix = logfluxs.shape[1]
        logA = np.mean(logfluxs, axis=1)
        lognormModels = logfluxs - logA[:, None]
        u,s,v = np.linalg.svd(lognormModels, full_matrices=False)
        # self.eigv0 = v

        if top is not None: 
            u = u[:, :top]
            s = s[:top]
            v = v[:top]
            self.pcaDim = len(v)
        nS = len(s)
        logging.info(f"Top #{nS} eigs {s[:10].round(2)}")
        assert np.allclose(v.dot(v.T), np.eye(nS))
        assert np.allclose(np.sum(v[:-1], axis=1).mean(), 0.0) 

        pcflux = u * s
        assert np.allclose(lognormModels.dot(v.T), pcflux)
        return logA, pcflux, v

#Obs --------------------------------------------------------------------------
    #TODO: fix it
    def init_obs_W(self, W, wave_H, step=None, flux_in_res=None):
        Obs = ObsW(W, step)
        Obs.init_W(wave_H, flux_in_res=flux_in_res)
        return Obs

    def init_obs(self):
        pass

# rand -------------------------------------------------------------------------

    def get_random_pmts_R(self, R, nPmt, nPara=5, method="halton", outRnd=False):
        rands = Util.get_random_uniform(nPmt, nPara, method=method, scaler=None)
        pmts = self.minmax_rescaler[R](rands)
        return [rands, pmts] if outRnd else pmts

    def get_random_pmts(self, nPmt, nPara=5, method="halton", outRnd=False):
        rands = Util.get_random_uniform(nPmt, nPara, method=method)
        pmts = self.minmax_rescaler(rands)
        return [rands, pmts] if outRnd else pmts

    @staticmethod
    def _get_random_pmt(PhyRng, PhyMin, N_pmt):
        pmt0 = np.random.uniform(0,1,(N_pmt,5))
        pmts = pmt0 * PhyRng + PhyMin   
        return pmts


# static ----------------------------------------------------------------------
    @staticmethod
    def init_para(para):
        return pd.DataFrame(para, columns=BaseBox.PhyShort)

    @staticmethod
    def get_bnd(R):
        bnd = np.array(BaseBox.DRs[R])
        PhyMin, PhyMax = bnd.T
        PhyRng = np.diff(bnd).T[0]
        PhyNum = PhyRng / BaseBox.PhyTick 
        PhyMid = (PhyNum //2) * BaseBox.PhyTick + PhyMin
        return PhyMin, PhyMax, PhyRng, PhyNum, PhyMid

    @staticmethod
    def get_minmax_scaler_fns(PhyMin, PhyRng):
        def scaler_fn(x):
            return (x - PhyMin) / PhyRng
        def inverse_scaler_fn(x):
            return x * PhyRng + PhyMin        
        return scaler_fn, inverse_scaler_fn

    @staticmethod
    def get_pdx_scaler_fns(PhyMin):
        def scaler_fn(x):
            return np.divide((x - PhyMin) ,BaseBox.PhyTick)
        def inverse_scaler_fn(x):
            return x * BaseBox.PhyTick + PhyMin
        return scaler_fn, inverse_scaler_fn

    @staticmethod
    def get_bdx_R(R, dfpara=None, bnds=None, cutCA = False):
        #TODO get range index
        if dfpara is None: dfpara = Util.IO.read_dfpara()
        if bnds is None: 
            bnds = BaseBox.DRs[R]
        Fs, Ts, Gs, Cs, As  = bnds

        maskM = (dfpara["M"] >= Fs[0]) & (dfpara["M"] <= Fs[1]) 
        maskT = (dfpara["T"] >= Ts[0]) & (dfpara["T"] <= Ts[1]) 
        maskL = (dfpara["G"] >= Gs[0]) & (dfpara["G"] <= Gs[1]) 
        mask = maskM & maskT & maskL
        if cutCA:
            maskC = (dfpara["C"] >= Cs[0]) & (dfpara["C"] <= Cs[1])
            maskA = (dfpara["A"] >= As[0]) & (dfpara["A"] <= As[1])
            mask = mask & maskC & maskA

        return dfpara[mask].index

    # def load_dfpara(self):
    #     dfpara = IO.read_dfpara()
    #     return dfpara

    @staticmethod
    def get_bdx(dfpara=None, para=None):
        if dfpara is None: 
            if para is None:
                dfpara = Util.IO.read_dfpara()
            else:
                dfpara = BaseBox.init_para(para)

        DBdx = {}
        for R in BaseBox.Rnms:
            bdx = BaseBox.get_bdx_R(R, dfpara, bnds=BaseBox.DRs[R], cutCA=False)
            DBdx[R] = bdx
        return DBdx


    @staticmethod
    def get_flux_para_R(R, flux, para, DBdx=None):
        if DBdx is None: DBdx = BaseBox.get_bdx(para=para)
        bdx = DBdx[R]
        boxFlux = flux[bdx]
        boxPara = para[bdx]
        logging.info("Flux {boxFlux.shape}, Para {boxPara.shape}")
        return bdx, boxFlux, boxPara

    @staticmethod
    def box_data(wave, flux, pdx, para, DBdx=None, Res=None, Rs=None, out=False, save=True):
        if save and (Res is None):
            raise ValueError("Res is None")
        if out:
            boxFluxs, boxPdxs, boxParas = {}, {} ,{}
        if DBdx is None: DBdx = BaseBox.get_bdx(para=para)
        if Rs is None: Rs = BaseBox.Rnms
        for R in Rs:
            boxFlux, boxPdx, boxPara = BaseBox.box_data_R(R, wave, flux, pdx, para, DBdx, Res, save)
            if out: 
                boxFluxs[R] = boxFlux
                boxPdxs[R] = boxPdx
                boxParas[R] = boxPara
        if out:
            return boxFluxs, boxPdxs, boxParas

    @staticmethod
    def box_data_R(R, wave, flux, pdx, para, DBdx=None, Res=None, save=False):
        bdx, boxFlux, boxPara = BaseBox.get_flux_para_R(R, flux, para, DBdx=DBdx)
        boxPdx = pdx[bdx] if pdx is not None else None
        RR = BaseBox.DRR[R]
        if save: Util.IO.save_bosz_box(Res, RR, wave, boxFlux, boxPdx, boxPara, overwrite=1)
        return boxFlux, boxPdx, boxPara

#collect ---------------------------------------------------------------
    @staticmethod
    def collect_fn(Rs, fn):
        Dfn_outs = {}
        for R in Rs:
            Dfn_outs[R] = fn(R)
        return Dfn_outs
