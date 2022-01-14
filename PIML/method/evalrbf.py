import os
import h5py
import numpy as np
from PIML import obs
from PIML.util.basebox import BaseBox
from PIML.util.baseplot import BasePlot

from PIML.method.rbf import RBF
import matplotlib.pyplot as plt
from tqdm import tqdm

from PIML.util.util import Util
from PIML.util.basespec import BaseSpec


class EvalRBF(BaseSpec):
    def __init__(self):
        super().__init__()
        self.Ws = [7100, 8500]
        self.R = None
        self.PhyMin = None
        self.PhyMax = None
        self.PhyRng = None
        self.PhyNum = None
        self.PhyMid = None
        self.PLT = BasePlot()
        self.topk = None



# init------------------------------------------------------------------------

    def init(self, Ws, R, Res, step=None, topk=10, onPCA=1):
        self.Ws = Ws
        self.R = R
        self.RR = Util.DRR[self.R]
        self.Res = Res
        self.onPCA = onPCA
        self.topk = topk
        wave_H, flux_H, self.pdx, self.para = self.Util.IO.load_bosz(Res, RR=self.RR)
        self.init_scaler()
        self.pdx0 = self.pdx - self.pdx[0]
        if step is not None:
            self.step = step
            self.wave_H, flux_H = self.get_flux_in_Wrange(wave_H, flux_H)
            self.wave, self.flux = self.downsample(flux_H)
        else:
            self.wave, self.flux = self.get_flux_in_Wrange(wave_H, flux_H)

        self.Npix = len(self.wave)
        self.flux0 = self.get_model(self.PhyMid, onGrid=1, plot=1)

        self.logflux = Util.safe_log(self.flux)
        self.rbf_flux = self.run_step_rbf(self.logflux, onPCA=onPCA)

    def init_scaler(self):
        self.PhyMin, self.PhyMax, self.PhyRng, self.PhyNum, self.PhyMid = BaseSpec.get_bnd_from_para(self.para)
        self.minmax_scaler, self.minmax_rescaler = BaseBox.get_minmax_scaler_fns(self.PhyMin, self.PhyRng)
        self.pmt2pdx_scaler, _ = BaseBox.get_pdx_scaler_fns(self.PhyMin)

    def get_flux_in_Wrange(self, wave, flux):
        return BaseSpec._get_flux_in_Wrange(wave, flux, self.Ws)    

    def downsample(self, flux_H):
        wave, flux = BaseSpec.resample(self.wave_H, flux_H, self.step)
        return wave, flux

#RBF------------------------------------------------------------------------
    def run_step_rbf(self, logflux, onPCA=1):
        self.RBF = RBF(coord=self.pdx0, coord_scaler=self.pmt2pdx_scaler)
        if onPCA:
            logA, pcflux = self.run_step_pca(logflux, top=self.topk)
            interp_AModel_fn, _, self.rbf_logA, self.rbf_ak = self.RBF.build_PC_rbf_interp(logA, pcflux, self.eigv)
            return interp_AModel_fn
        else:
            interp_flux_fn = self.RBF.build_logflux_rbf_interp(logflux)
            return interp_flux_fn

    def run_step_pca(self, logfluxs, top=10):

        logA = np.mean(logfluxs, axis=1)
        lognormModels = logfluxs - logA[:, None]

        u,s,v = np.linalg.svd(lognormModels, full_matrices=False)
        
        self.eigv0 = v
        if top is not None: 
            u = u[:, :top]
            s = s[:top]
            v = v[:top]
            self.topk = top
        print("Top10 eigs", s[:10].round(2))
        assert abs(np.mean(np.sum(v.dot(v.T), axis=0)) -1) < 1e-5
        assert abs(np.sum(v, axis=1).mean()) < 0.1
        self.eigv = v
        pcflux = u * s
        assert (lognormModels.dot(v.T) - pcflux).max() < 1e-5
        return logA, pcflux

    def test_rbf(self, pmt1, pmt2, pmt=None):
        flux1, flux2 = self.get_model(pmt1,onGrid=1),  self.get_model(pmt2,onGrid=1)
        if pmt is None: pmt = 0.5 * (pmt1 + pmt2)
        interpFlux = self.rbf_flux(pmt)
        plt.plot(self.wave, flux1, label=Util.get_pmt_name(*pmt1), c='k')
        plt.plot(self.wave, interpFlux, label=Util.get_pmt_name(*pmt), c='r')
        plt.plot(self.wave, flux2, label=Util.get_pmt_name(*pmt2), c='b')
        plt.legend()


# model------------------------------------------------------------------------
    def get_model(self, pmt, norm=0, onGrid=0, plot=0):
        if norm:
            model= self.interp_model_fn(pmt)
            if plot: BasePlot.plot_spec(self.wave, model, pmt=pmt)
            return model
        else:
            if onGrid:
                fdx = BaseSpec.get_fdx_from_pmt(pmt, self.para)
                Amodel = self.flux[fdx]
            else:
                Amodel = self.interp_flux_fn(pmt)
            if plot: BasePlot.plot_spec(self.wave, Amodel, pmt=pmt)
            return Amodel
    



