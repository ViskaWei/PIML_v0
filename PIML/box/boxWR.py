
import numpy as np
from PIML.util.basebox import BaseBox
from PIML.util.baseplot import BasePlot

from PIML.obs.obs import Obs
from PIML.method.llh import LLH
from PIML.method.rbf import RBF
import matplotlib.pyplot as plt
from tqdm import tqdm


class BoxWR(BaseBox):
    def __init__(self):
        super().__init__()
        self.W = None
        self.R = None
        self.PhyMin = None
        self.PhyMax = None
        self.PhyRng = None
        self.PhyNum = None
        self.PhyMid = None
        self.Obs = Obs()
        self.RBF = RBF()
        self.LLH = LLH()
        self.PLT = BasePlot()
        np.random.seed(1015)

# init------------------------------------------------------------------------

    def init_R(self, R):
        self.R = R
        self.RR = BaseBox.DRR[R]
        self.PhyMin, self.PhyMax, self.PhyRng, self.PhyNum, self.PhyMid = self.get_bnd(R)
        self.scaler, self.rescaler = BaseBox.get_scaler_fns(self.PhyMin, self.PhyRng)
        self.pmt2pdx_scaler, _ = BaseBox.get_pdx_scaler_fns(self.PhyMin)

    def init_W(self, W):
        self.W = W
        self.Ws = Obs.init_W(self.W)

    def get_flux_in_Wrange(self, wave, flux):
        return Obs._get_flux_in_Wrange(wave, flux, self.Ws)

    def init(self, W, R, Res, step, onPCA=1):
        self.init_W(W)
        self.init_R(R)
        self.init_plot()
        self.Res = Res
        self.step = step
        wave_H, flux_H, self.pdx, self.para = self.IO.load_bosz(Res, RR=self.RR)
        self.pdx0 = self.pdx - self.pdx[0]

        self.wave_H, flux_H = self.get_flux_in_Wrange(wave_H, flux_H)
        self.wave, self.flux = self.downsample(flux_H)
        
        self.flux0 = self.get_model(self.PhyMid, onGrid=1, plot=1)
        self.init_sky(self.wave_H, self.flux0, step)

        self.logflux = self.Obs.safe_log(self.flux)
        self.interp = self.run_step_rbf(self.logflux, onPCA=onPCA)
        self.init_LLH()

    def init_plot(self):
        self.PLT.make_box_fn = lambda x: self.PLT.box_fn(self.PhyRng, self.PhyMin, 
                                                        self.PhyMax, n_box=x, 
                                                        c=BoxWR.DRC[self.R], RR=self.RR)


    def init_sky(self, wave_H, flux, step):
        print(wave_H.shape, flux.shape)

        self.Obs.prepare_sky(wave_H, flux, step)

    def init_LLH(self): 
        self.LLH.get_model = lambda x: self.get_model(x, onGrid=0, plot=0)
        self.LLH.x0 = self.PhyMid
        self.LLH.PhyMin = self.PhyMin
        self.LLH.PhyMax = self.PhyMax

    def downsample(self, flux_H):
        wave, flux = Obs.resample(self.wave_H, flux_H, self.step)
        return wave, flux


    def run_step_rbf(self, logflux, onPCA=1):
        if onPCA:
            mu, pcflux = self.run_step_pca(logflux)
            interp_fn = self.build_PC_rbf(mu, pcflux)
        else:
            interp_fn = self.build_logflux_rbf(logflux)
        return interp_fn

    def build_logflux_rbf(self, logflux):
        rbf_interp = self.RBF.train_rbf(self.pdx0, logflux)
        rbf = self.RBF.build_rbf(rbf_interp, self.pmt2pdx_scaler, np.exp)
        return rbf

    def build_PC_rbf(self, mu, pcflux):
        rbf_interp_mu = self.RBF.train_rbf(self.pdx0, mu)
        rbf_mu = self.RBF.build_rbf(rbf_interp_mu, self.pmt2pdx_scaler, None)
        rbf_interp_coeff = self.RBF.train_rbf(self.pdx0, pcflux)
        rbf_coeff = self.RBF.build_rbf(rbf_interp_coeff, self.pmt2pdx_scaler, None)
        def interp_fn(x):
            mu = rbf_mu(x)
            coeff = rbf_coeff(x)
            print(mu.shape, coeff.shape)
            logflux = coeff.dot(self.eigv) + mu
            return np.exp(logflux)
        return interp_fn

    def run_step_pca(self, logflux, top=10):

        mu = np.mean(logflux, axis=1)
        normflux = logflux - mu[:, None]

        u,s,v = np.linalg.svd(normflux, full_matrices=False)
        
        self.eigv0 = v
        if top is not None: 
            u = u[:, :top]
            s = s[:top]
            v = v[:top]
        print(s[:10].round(2))
        assert abs(np.mean(np.sum(v.dot(v.T), axis=0)) -1) < 1e-5
        assert abs(np.sum(v, axis=1).mean()) < 0.1
        self.eigv = v
        pcflux = u * s
        assert (normflux.dot(v.T) - pcflux).max() < 1e-5
        return mu, pcflux

    def test_rbf(self, pmt1, pmt2, pmt=None):
        flux1, flux2 = self.get_model(pmt1,onGrid=1),  self.get_model(pmt2,onGrid=1)
        if pmt is None: pmt = 0.5 * (pmt1 + pmt2)
        interpFlux = self.interp([pmt])[0]
        plt.plot(self.wave, interpFlux, label= pmt)
        plt.plot(self.wave, flux1, label = pmt1)
        plt.plot(self.wave, flux2, label = pmt2)
        plt.legend()

# model------------------------------------------------------------------------
    def get_model(self, pmt, onGrid=0, plot=0):
        if onGrid:
            fdx = self.Obs.get_fdx_from_pmt(pmt, self.para)
            flux = self.flux[fdx]
        else:
            flux = self.interp([pmt])[0]
        if plot: self.Obs.plot_spec(self.wave, flux, pmt=pmt)
        return flux
    

    def make_obs_from_pmt(self, pmt, snr, N=1, plot=0):
        noise_level = self.Obs.snr2nl(snr)
        flux = self.get_model(pmt)
        if N==1:
            obsflux, obsvar = self.Obs.add_obs_to_flux(flux, noise_level)
            if plot: self.Obs.plot_noisy_spec(self.wave, flux, obsflux, pmt)
        else:
            obsflux, obsvar = self.Obs.add_obs_to_flux_N(flux, noise_level, N)
        return obsflux, obsvar
    

#LLH --------------------------------------------------


    def eval_LLH_at_pmt(self, pmt, pdxs=[1], snr=10, N_obs=10, plot=0):
        obsfluxs , obsvar = self.make_obs_from_pmt(pmt, snr, N=N_obs)
        snr = self.Obs.get_avg_snr(obsfluxs)
        fns = self.LLH.get_eval_LLH_fns(pdxs, pmt, obsvar)
        preds = []
        for fn_pdx, fn in fns.items():
            pred_x = self.LLH.collect_estimation(fn, obsfluxs, fn_pdx, x0=self.PhyMid)
            preds.append(pred_x)
        preds = np.array(preds).T
        if plot:
            self.plot_eval_LLH(pmt, preds, pdxs, snr)
        return preds, snr

    def eval_LLH_snr(self, snr, pmts=None, pdxs=[0,1,2], N_pmt=10, N_obs=10, n_box=0.5):
        if pmts is None: pmts = self.get_random_grid_pmt(N_pmt)
        fns = []
        SNRs= []
        for pmt in tqdm(pmts):
            preds_pmt, SNR = self.eval_LLH_at_pmt(pmt, pdxs, snr=snr, N_obs=N_obs, plot=0)
            fns_pmt = self.PLT.flow_fn_i(preds_pmt, pmt[pdxs], legend=0)
            fns = fns + fns_pmt
            SNRs.append(SNR)

        f = self.PLT.plot_box(pdxs, fns = fns, n_box=n_box)
        f.suptitle(f"SNR={SNR.mean():.2f}")

    def eval_LLH(self, pmts=None, pdxs=[0,1,2], N_pmt=10, n_box=0.5, snrList=None):
        if snrList is None: snrList = self.Obs.snrList
        # if pmts is None: pmts = self.get_random_grid_pmt(N_pmt)
        for snr in snrList:
            self.eval_LLH_snr(snr, pmts, pdxs, N_pmt, n_box)

    def plot_eval_LLH(self, pmt, pred, pdxs, snr, n_box=0.5):
        fns = self.PLT.flow_fn_i(pred, pmt[pdxs], snr, legend=0)
        f = self.PLT.plot_box(pdxs, fns = fns, n_box=n_box)
        f.suptitle(f"SNR = {snr}")

    def get_random_grid_pmt(self, N_pmt):
        np.random.seed(42)
        idx = np.random.randint(0, len(self.para), N_pmt)
        pmts = self.para[idx]
        return pmts