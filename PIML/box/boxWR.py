
import numpy as np
from PIML.util.basebox import BaseBox
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

# init------------------------------------------------------------------------

    def init_R(self, R):
        self.R = R
        self.RR = BaseBox.DRR[R]
        self.PhyMin, self.PhyMax, self.PhyRng, self.PhyNum, self.PhyMid = self.get_bnd(R)
        self.scaler, self.rescaler = BaseBox.get_scaler_fns(self.PhyMin, self.PhyRng)
        self.rbf_scaler, _ = BaseBox.get_pdx_scaler_fns(self.PhyMin)

    def init_W(self, W):
        self.W = W
        self.Ws = Obs.init_W(self.W)

    def get_flux_in_Wrange(self, wave, flux):
        return Obs._get_flux_in_Wrange(wave, flux, self.Ws)

    def init(self, W, R, Res, step):
        self.init_W(W)
        self.init_R(R)
        self.Res = Res
        self.step = step
        wave_H, flux_H, self.pdx, self.para = self.IO.load_bosz(Res, RR=self.RR)
        self.pdx0 = self.pdx - self.pdx[0]

        self.wave_H, flux_H = self.get_flux_in_Wrange(wave_H, flux_H)
        self.wave, self.flux = self.downsample(flux_H)
        
        self.flux0 = self.get_model(self.PhyMid, onGrid=1, plot=1)
        self.init_sky(self.wave_H, self.flux0, step)

        self.build_rbf(self.flux)
        self.init_LLH()

    def init_sky(self, wave_H, flux, step):
        self.Obs.prepare_sky(wave_H, flux, step)

    def init_LLH(self): 
        self.LLH.get_model = lambda x: self.get_model(x, onGrid=0, plot=0)
        self.LLH.x0 = self.PhyMid
        self.LLH.PhyMin = self.PhyMin
        self.LLH.PhyMax = self.PhyMax

    def downsample(self, flux_H):
        wave, flux = Obs.resample(self.wave_H, flux_H, self.step)
        return wave, flux


    def build_rbf(self, flux):
        logflux = self.Obs.safe_log(flux)
        self.RBF._build_rbf(self.pdx0, logflux)
        self.RBF.rbf_scaler = self.rbf_scaler
        self.RBF.pred_scaler = np.exp

    def interp(self, pmt):
        pmt = np.array(pmt)
        if len(pmt.shape) == 1:
            return self.RBF.rbf_predict([pmt])[0]
        else:
            return self.RBF.rbf_predict(pmt)

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
            flux = self.interp(pmt)
        if plot: self.Obs.plot_spec(self.wave, flux, pmt=pmt)
        return flux
    

    def make_obs_from_pmt(self, pmt, snr, N=1, plot=0):
        noise_level = self.Obs.snr_from_nl(snr)
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
        fns = self.LLH.get_eval_LLH_fns(pdxs, pmt, obsvar)
        preds = []
        for fn_pdx, fn in fns.items():
            pred_x = self.LLH.collect_estimation(fn, obsfluxs, fn_pdx, x0=self.PhyMid)
            preds.append(pred_x)
        preds = np.array(preds).T
        if plot:
            self.plot_eval_LLH(pmt, preds, pdxs, snr)
        return preds

    def eval_LLH_NL(self, noise_level, pmts=None, pdxs=[0,1,2], N_pmt=10, n_box=0.5):
        if pmts is None: pmts = self.get_random_pmt(N_pmt)
        fns = []
        for pmt in tqdm(pmts):
            preds_pmt = self.eval_LLH_at_pmt(pmt, pdxs, noise_level=noise_level, N_obs=100, plot=0)
            fns_pmt = self.flow_fn_i(preds_pmt, pmt[pdxs], legend=0)
            fns = fns + fns_pmt

        f = self.plot_box(pdxs, fns = fns, n_box=n_box)
        f.suptitle(f"NL={noise_level}")

    def eval_LLH(self, pmts=None, pdxs=[0,1,2], N_pmt=10, n_box=0.5):
        if pmts is None: pmts = self.get_random_pmt(N_pmt)
        for NL in [1,10,50,100]:
            self.eval_LLH_NL(NL, pmts, pdxs, N_pmt, n_box)

    def plot_eval_LLH(self, pmt, pred, pdxs, snr, n_box=0.5):
        fns = self.flow_fn_i(pred, pmt[pdxs], snr, legend=0)
        f = self.plot_box(pdxs, fns = fns, n_box=n_box)
        f.suptitle(f"SNR = {snr}")

    def get_random_grid_pmt(self, N_pmt):
        idx = np.random.randint(0, len(self.para), N_pmt)
        pmts = self.para[idx]
        return pmts