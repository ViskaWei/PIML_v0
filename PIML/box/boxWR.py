import os
import h5py
import numpy as np
from PIML import obs
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
        self.topk = None



# init------------------------------------------------------------------------

    def get_flux_in_Wrange(self, wave, flux):
        return Obs._get_flux_in_Wrange(wave, flux, self.Ws)

    def init(self, W, R, Res, step, topk=10, onPCA=1):
        self.init_WR(W, R)
        self.init_plot()
        self.Res = Res
        self.step = step
        self.onPCA = onPCA
        self.topk = topk
        wave_H, flux_H, self.pdx, self.para = self.IO.load_bosz(Res, RR=self.RR)
        self.pdx0 = self.pdx - self.pdx[0]
        self.wave_H, flux_H = self.get_flux_in_Wrange(wave_H, flux_H)
        # self.flux_H = flux_H
        self.Mdx = self.Obs.get_fdx_from_pmt(self.PhyMid, self.para)
        self.fluxH0 = flux_H[self.Mdx]

        self.wave, self.flux = self.downsample(flux_H)
        self.Npix = len(self.wave)
        self.flux0 = self.get_model(self.PhyMid, onGrid=1, plot=1)

        self.init_sky(self.wave_H, self.flux0, step)

        self.logflux = self.Obs.safe_log(self.flux)
        self.interp_obs_fn, self.interp_model_fn, self.interp_stdmag_fn, self.interp_bias_fn = self.run_step_rbf(self.logflux, onPCA=onPCA, Obs=self.Obs)
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


    def run_step_rbf(self, logflux, onPCA=1, Obs=None):
        if onPCA:
            logA, pcflux = self.run_step_pca(logflux, top=self.topk)
            interp_obs_fn, interp_model_fn, interp_stdmag_fn, interp_bias_fn = self.build_PC_rbf(logA, pcflux, Obs)
            return interp_obs_fn, interp_model_fn, interp_stdmag_fn, interp_bias_fn
        else:
            interp_fn = self.build_logflux_rbf(logflux)
            return interp_fn, interp_fn, None, None

    def build_logflux_rbf(self, logflux):
        rbf_interp = self.RBF.train_rbf(self.pdx0, logflux)
        rbf = self.RBF.build_rbf(rbf_interp, self.pmt2pdx_scaler, np.exp)
        return rbf

    def build_PC_rbf(self, logA, pcflux, Obs=None):
        rbf_interp_logA = self.RBF.train_rbf(self.pdx0, logA)
        rbf_logA = self.RBF.build_rbf(rbf_interp_logA, self.pmt2pdx_scaler, None)
        self.rbf_logA = rbf_logA
        rbf_interp_ak = self.RBF.train_rbf(self.pdx0, pcflux)
        rbf_ak = self.RBF.build_rbf(rbf_interp_ak, self.pmt2pdx_scaler, None)
        self.rbf_ak = rbf_ak
        
        def interp_obs_fn(pmt, log=0):
            logA = rbf_logA(pmt)
            ak = rbf_ak(pmt)
            logModel = ak.dot(self.eigv)
            logAModel = logModel + logA
            if log: 
                return logAModel
            else:
                return np.exp(logAModel)

        def interp_model_fn(pmt, log=0):
            ak = rbf_ak(pmt)
            logModel = ak.dot(self.eigv)
            if log: 
                return logModel
            else:
                return np.exp(logModel)

        def interp_stdmag_fn(pmt, noise_level):
            if Obs is not None:
                AModel = interp_obs_fn(pmt, log=0)
                var_in_res = Obs.get_var(AModel, Obs.sky_in_res, step=self.step)
                sigma_in_res = np.sqrt(var_in_res)
                stdmag = np.divide(noise_level * sigma_in_res, AModel)
                return stdmag
            else:
                return None
        
        def interp_bias_fn(stdmag, X=None):
            if Obs is not None:
                if X is None: X = np.random.normal(0,1, self.Npix)
                bias = self.eigv.dot(np.log(1 + np.multiply(X, stdmag)))
                return bias
            else:
                return None

        return interp_obs_fn, interp_model_fn, interp_stdmag_fn, interp_bias_fn

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
        interpFlux = self.interp_obs_fn(pmt)
        plt.plot(self.wave, interpFlux, label= pmt)
        plt.plot(self.wave, flux1, label = pmt1)
        plt.plot(self.wave, flux2, label = pmt2)
        plt.legend()

# model------------------------------------------------------------------------
    def get_model(self, pmt, norm=0, onGrid=0, plot=0):
        if norm:
            model= self.interp_model_fn(pmt)
            if plot: self.Obs.plot_spec(self.wave, model, pmt=pmt)
            return model
        else:
            if onGrid:
                fdx = self.Obs.get_fdx_from_pmt(pmt, self.para)
                Amodel = self.flux[fdx]
            else:
                Amodel = self.interp_obs_fn(pmt)
            if plot: self.Obs.plot_spec(self.wave, Amodel, pmt=pmt)
            return Amodel
    
    def get_bk_fn_from_pmt(self, pmt, noise_level):
        ak = self.rbf_ak(pmt)
        stdmag = self.interp_stdmag_fn(pmt, noise_level)
        def add_bias(X=None):
            bias = self.interp_bias_fn(stdmag, X)
            return ak + bias
        return add_bias

    def get_bk_fns(self, noise_level, pmts=None, N_pmts=1, out_bks=1):
        if pmts is None: 
            pmts = self.get_random_pmt(N_pmts)
        if out_bks:
            bks = np.zeros((pmts.shape[0], self.topk))
        bk_fns = []
        for ii, pmt in enumerate(pmts):
            bk_fn = self.get_bk_fn_from_pmt(pmt, noise_level)
            bk_fns.append(bk_fn)
            if out_bks:
                bks[ii] = bk_fn()            
        if out_bks:
            return bk_fns, bks
        else:
            return bk_fns
        
    def get_bks(self, bk_fns):
        bks = np.zeros((len(bk_fns), self.topk))
        for ii, bk_fn in enumerate(bk_fns):
            bks[ii] = bk_fn()
        return bks

    def get_bks_N_obs_from_pmt(self, pmt=None, noise_level=1, N_obs=1):
        if pmt is None: pmt = self.get_random_pmt(1)[0]
        bk_fn = self.get_bk_fn_from_pmt(pmt, noise_level)
        bk_N_obs = np.zeros((N_obs, self.topk))
        for ii in range(N_obs):
            bk_N_obs[ii] = bk_fn()
        return bk_N_obs
        


    def make_obs_from_pmt(self, pmt, noise_level=None, snr=None, N=1, plot=0, onPCA=0, onGrid=1):
        if noise_level is None:
            if snr == np.inf: 
                noise_level = 0
            else:
                noise_level = self.Obs.snr2nl(snr)
        AModel = self.get_model(pmt, norm=0, onGrid=0)
        # np.random.seed(1015)
        obsfluxs, obsvar = self.Obs.add_obs_to_flux_N(AModel, noise_level, self.step, N)
        if plot: self.Obs.plot_noisy_spec(self.wave, AModel, obsfluxs[0], pmt)
        # if onPCA:
        #     normflux = self.get_model(pmt, norm=1)
        #     A = np.exp(self.rbf_logA(pmt))
        #     A0 = obsfluxs.mean(1).mean() / normflux.mean() 
        #     print(f"A = {A}, dA = {A-A0}")
        #     bias = self.get_bias(normflux, obsvar, A=A)
            
        #     # print(normflux - flux)
        #     if N == 1: obsfluxs = obsfluxs[0]
        #     return obsfluxs.mean(0), obsvar, bias, A
        else:
            if N == 1: obsfluxs = obsfluxs[0]
            return obsfluxs, obsvar

    def estimate_snr(self, NL):
        obsfluxH0, _ = self.Obs.add_obs_to_flux(self.fluxH0, NL, step=0)
        bosz_5000_snr_factor = np.sqrt(2)
        snr = self.Obs.get_snr(obsfluxH0) / bosz_5000_snr_factor
        return snr

    def eval_pca_bias(self, pmt, N, noise_level=None, snr=None):
        if (noise_level is None):
            if (snr is None): 
                raise "noise level or snr not specified"
            else:
                noise_level= self.Obs.snr2nl(snr)
        obsfluxs, obsvar = self.make_obs_from_pmt(pmt, noise_level=noise_level,snr=snr, N=N, onPCA=1, plot=0)
        obssigma = np.sqrt(obsvar)
        AModel = self.interp_obs_fn(pmt, log=0)
        nu = (obsfluxs - AModel)
        bk = np.log(obsfluxs).dot(self.eigv.T)
        X = nu / obssigma
        X2 = X**2
        if N > 1: 
            nu = nu.mean(0)
            bk = bk.mean(0)
            X0 = X.mean(0)
            X20 = X2.mean(0)
        ak = self.rbf_ak(pmt)
        stdmag = self.interp_stdmag_fn(pmt, noise_level)
        varmag = X20 * stdmag**2
        bias = self.get_bias_from_stdmag(stdmag)
        biasX= self.get_bias_from_stdmag(np.multiply(X0, stdmag),  varmag)
        
        return bk - ak, bias, biasX

    def get_bias_from_stdmag(self, stdmag, varmag=None):
        bias_all = self.eigv.dot(np.log(1 + stdmag))
        bias_1st_order = self.eigv.dot(stdmag)
        if varmag is None: varmag = stdmag ** 2
        bias_2nd_order = 1/2 * self.eigv.dot(varmag)
        return bias_all, bias_1st_order, bias_2nd_order


    def plot_theory_bias(self, ak, bias, NL=None, ax=None, pmt=None, log=1, theory=1, N=None, lgd=0):
        if ax is None: 
            f, ax = plt.subplots(1, figsize=(6,4), facecolor="w")
        b0, b1, b2 = bias
        ak = abs(ak)
        b0 = abs(b0)
        b1 = abs(b1)
        b2 = abs(b2)
        Nname = f"N={N}" if N is not None else ""
        if theory:
            labels =   ["$|\sum_p \log(1+ \sigma_p / A m_p) \cdot V_p|$",  
                        "$|\sum_p (\sigma_p/ A m_p)  \cdot  V_p|$", 
                        "$\sum_p 1/2 \cdot (\sigma_p / A m_p)^2 \cdot V_p$",
                        "Theory $|a_k|$ "+Nname]
        else:
            labels =   [r"$|\sum_p \log(1+ \nu_p / A m_p) \cdot V_p|$",  
                        r"$|\sum_p (\nu_p/ A m_p)  \cdot  V_p|$", 
                        r"$|\sum_p 1/2 \cdot (\nu_p / A m_p)^2 \cdot V_p|$",
                        "$|a_k|$" + Nname]
        ax.plot(ak, b0, 'ko', label=labels[0])
        ax.plot(ak, b1, 'rx', label=labels[1])
        ax.plot(ak, b2, 'bo', label=labels[2])
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel(labels[3])
        # ax.set_xlim(1e-7, 1e-2)
        ax.set_ylim(1e-7, 1e-2)

        if pmt is None: pmt = self.PhyMid
        title = self.RR + " " + self.Obs.get_pmt_name(*pmt)
        ax.set_title(title)
        if lgd: ax.legend(bbox_to_anchor=(0.55, 0.5),  ncol=1)
        # ax.legend()
        return b0
    


    def plot_exp_bias(self, ak, diffs, labels=None, ax=None, pmt=None):
        
        if ax is None: 
            f, ax = plt.subplots(1, figsize=(6,4), facecolor="w")
        ak0 = abs(ak)
        for ii, diff in enumerate(diffs):
            d0 = abs(diff) 
            ax.plot(ak0, d0, 'o', label=labels[ii])
            # ax.plot(ak0, d0 / ak0, 'o', label=labels[ii])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("|$a_k$|")
        ax.set_ylabel("|$b_k$ - $a_k$|")

        # ax.set_ylabel("| $b_k$ / $a_k$ - 1|")
        if pmt is None: pmt = self.PhyMid
        title = self.RR + " " + self.Obs.get_pmt_name(*pmt)
        ax.set_title(title)
        ax.legend()

    def plot_bias_evals(self, diffs, bias, biasX =None, pmt=None, diff_labels=None, N=None):
        if pmt is None: pmt = self.PhyMid
        f, axs = plt.subplots(1, 3, figsize=(16,4), facecolor="w")
        ak = self.rbf_ak(pmt)
        self.plot_theory_bias(ak, bias, ax=axs[0], N=N)
        if biasX is not None:
            self.plot_theory_bias(ak, biasX, ax=axs[1], log=1, theory=0, N=N)
        ak = self.rbf_ak(pmt)
        self.plot_exp_bias(ak, diffs, labels=diff_labels, ax=axs[2])




    def get_bias(self, flux, obsvar, A=1):
        x = np.divide(obsvar**0.5, A * flux)
        bias1 = self.eigv.dot(x)
        bias_all = self.eigv.dot(np.log(1+x))
        bias2 = 0.5 * self.eigv.dot(np.divide(obsvar, A**2 * flux**2))
        return bias1, bias2, bias_all
        
    def eval_ak_bias(self, pmt, snr=10, N_obs=10, plot=0, N_plot=None):
        AK = self.rbf_ak(pmt)
        obsfluxs, obsvar, bias, A = self.make_obs_from_pmt(pmt, snr, N=N_obs, onPCA = self.onPCA)
        logobsflux = self.Obs.safe_log(obsfluxs)
        ak = logobsflux.dot(self.eigv.T)
        if snr !=np.inf: 
            snr = self.Obs.get_avg_snr(obsfluxs, top = N_obs)
        if plot: 
            if N_plot is None: N_plot = self.topk // 2 -1
            self.plot_eval_ak_bias(AK, ak, pmt, snr, N_plot = N_plot)
        print(f"diff = {AK.mean(0) - ak - bias[1]}")
        print(f"diff_log(1+x) = {AK.mean(0) - ak - bias[2]}")

        return AK, ak, bias, snr, obsfluxs.mean(0), obsvar,A

    def save_ak(self, pmt=None, SAVE_PATH=None):
        if pmt is None: pmt = self.PhyMid
        if SAVE_PATH is None: SAVE_PATH = os.path.join(self.Obs.DATA_PATH, "ak.h5")
        ak= self.rbf_ak(pmt)
        with h5py.File(SAVE_PATH, "a") as f:
            f.create_dataset(self.R, data=ak, shape=ak.shape)

    def load_ak(self, pmt=None, LOAD_PATH=None):
        if LOAD_PATH is None: LOAD_PATH = os.path.join(self.Obs.DATA_PATH, "ak.h5")
        Dak = {}
        with h5py.File(LOAD_PATH, "r") as f:
            for key in f.keys():
                Dak[key] = f[key][:]
        return Dak

    def plot_Dak(self, Dak):
        f, axs = plt.subplots(3,2, figsize=(16,12), facecolor="w")
        axs = axs.flat
        for ii, (key, val) in enumerate(Dak.items()):
            axs[ii].plot(Dak[key], 'ro', label=key)
            axs[ii].set_xlabel("$a_k$")

        # ax.set_title(f"{self.RR} {self.Obs.get_pmt_name(*pmt)}")
    
#LLH --------------------------------------------------


    def eval_LLH_at_pmt(self, pmt, odxs=[1], noise_level=100, N_obs=10, plot=0):
        obsfluxs , obsvar = self.make_obs_from_pmt(pmt, noise_level=noise_level, N=N_obs)
        # if snr !=np.inf: 
        #     snr = self.Obs.get_avg_snr(obsfluxs, top = N_obs)
        fns = self.LLH.get_eval_LLH_fns(odxs, pmt, obsvar)
        preds = []
        for fn_pdx, fn in fns.items():
            pred_x = self.LLH.collect_estimation(fn, obsfluxs, fn_pdx, x0=self.PhyMid)
            preds.append(pred_x)
        preds = np.array(preds).T
        if plot:
            snr = self.estimate_snr(noise_level)
            self.plot_eval_LLH(pmt, preds, odxs, snr)
        return preds


    def plot_eval_ak_bias(self, pred, truth, pmt, snr, N_plot, n_box=0.5):
        fns = self.PLT.flow_fn_i(pred, truth, snr, legend=0)
        f = self.PLT.plotN(N_plot=N_plot, fns = fns, lbl="PC - a")
        name = self.Obs.get_pmt_name(*pmt)
        f.suptitle(f'{name} || snr {snr:.1f}')
        f.tight_layout()



    def plot_ak_cdxs(self, AK, ak, pmt, snr, N=3):
        N = np.min([N, len(ak)])
        f, axs = plt.subplots(1,N, figsize=(N*3,3), facecolor="w")
        for ii in range(N):
            ax = axs[ii]
            self.plot_ak_cdx(ak, AK, 2*ii, 2*ii + 1, ax=ax)
        name = self.Obs.get_pmt_name(*pmt)
        f.suptitle(f'{name} || snr {snr:.1f}')
        f.tight_layout()

    def eval_ak_snr(self, snr, pmts=None, N_pmt=10, N_obs=10, N_plot=None):
        if pmts is None: pmts = self.get_random_grid_pmt(N_pmt)
        if N_plot is None: N_plot = self.topk // 2 - 1
        fns = []
        SNRs= []
        for pmt in tqdm(pmts):
            AK, ak, bias, SNR = self.eval_ak_bias(pmt, snr=snr, N_obs=N_obs, plot=0, N_plot=N_plot)
            fns_pmt = self.PLT.flow_fn_i(AK, ak, legend=0)
            fns = fns + fns_pmt
            SNRs.append(SNR)
        f = self.PLT.plotN(N_plot=N_plot, fns = fns, lbl="PC - a")
        name = self.Obs.get_pmt_name(*pmt)
        f.suptitle(f'{name} || snr {snr:.1f}')
        f.tight_layout()


    def eval_ak(self, snrList=[50], pmts=None, N_pmt=10, N_obs=10, N_plot=None):
        if pmts is None: pmts = self.get_random_grid_pmt(N_pmt)
        if N_plot is None: N_plot = self.topk // 2 - 1
        fns = {}
        SNRs = {}
        N_snr = len(snrList)
        fig, axss = plt.subplots(N_snr, N_plot, figsize=(N_plot*3, N_snr*3), facecolor="w", sharex="col", sharey="row")
        
        for ii, snr in enumerate(snrList):
            fns_snr = []
            SNRs_snr= []
            axs = axss[ii]
            for pmt in tqdm(pmts):
                AK, ak, bias, SNR = self.eval_ak_bias(pmt, snr=snr, N_obs=N_obs, plot=0, N_plot=N_plot)
                fns_pmt = self.PLT.flow_fn_i(AK, ak, legend=0)
                fns_snr = fns_snr + fns_pmt
                SNRs_snr.append(SNR)
            fns[snr] = fns_snr
            SNR_snr = np.mean(SNRs_snr)
            SNRs[snr] = SNR_snr
            self.PLT.plotN(N_plot=N_plot, fns = fns_snr, lbl="PC - a", axs=axs)
            axs[0].set_title(f'SNR {SNR_snr:.1f}')
        name = self.Obs.get_pmt_name(*pmt)
        fig.suptitle(f'{name}')
        fig.tight_layout()
        return fns, SNRs, pmts

    def plot_ak_cdx(self, ak, AK, cdx1, cdx2, ax=None):
        if ax is None: fig, ax = plt.subplots(1,1)
        data1, data2 = AK[:,cdx1], AK[:,cdx2]
        ax.plot(data1, data2, "o", color="gray", markersize=2)
        ax.plot(ak[cdx1], ak[cdx2], "ro", label="$a_k$")
        ax.plot(data1.mean(), data2.mean(), "go", label="$b_k$")
        ax.legend()

        ax.set_xlabel(f"PC - a{cdx1}")
        ax.set_ylabel(f"PC - a{cdx2}") 


    def eval_LLH_NL(self, noise_level=None, pmts=None, pdxs=[0,1,2], N_pmt=10, N_obs=10, n_box=0.5):
        if pmts is None: pmts = self.get_random_pmt(N_pmt)
        fns = []
        for pmt in tqdm(pmts):
            preds_pmt = self.eval_LLH_at_pmt(pmt, pdxs, noise_level=noise_level, N_obs=N_obs, plot=0)
            fns_pmt = self.PLT.flow_fn_i(preds_pmt, pmt[pdxs], legend=0)
            fns = fns + fns_pmt

        f = self.PLT.plot_box(pdxs, fns = fns, n_box=n_box)
        snr = self.estimate_snr(noise_level)
        f.suptitle(f"SNR={snr:.2f}")

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

    def get_random_pmt(self, N_pmt, nPara=5):
        pmt0 = np.random.uniform(0,1,(N_pmt,nPara))
        pmts = self.minmax_rescaler(pmt0)
        return pmts

    def plot_ak_on_pmt(self, fn, topk=10, axis="T", fn_name=""):
        adx = BoxWR.PhyShort.index(axis)
        pmt = np.copy(self.PhyMid)
        x = pmt[adx]
        def fn_adx(X):
            pmt[adx] = X
            return fn(pmt)
        x_large = np.linspace(self.PhyMin[adx], self.PhyMax[adx], 101)
        x_small = np.linspace(x - BoxWR.PhyTick[adx], x + BoxWR.PhyTick[adx], 25)
        y1 = []
        y2 = []
        for xi in x_large:
            y1.append(fn_adx(xi))
        y1 = np.array(y1).T
        for xj in x_small:
            y2.append(fn_adx(xj))
        y2 = np.array(y2).T
        fn_x_val = fn_adx(x)
        # MLE_X = -1 * fn(X)
        plt.figure(figsize=(15,6), facecolor="w")
        for ii in range(topk):
            plt.plot(x_large, y1[ii], "o",markersize=7, label = f"PC-a{ii}")    
            # plt.plot(x,       fn_x_val[ii], 'ro', label=f"Truth {fn_x_val[ii]:.2f}")
            plt.plot(x,       fn_x_val[ii], 'ro')

        # plt.plot(X, MLE_X, 'ko', label=f"Estimate {MLE_X:.2f}")
        xname = BoxWR.PhyLong[adx]
        ts = ""
        # ts = f'{xname} Truth={x:.2f}K, {xname} Estimate={X:.2f}K, S/N={SN:3.1f}'
        # ts = ts +  'sigz={:6.4f} km/s,  '.format(np.sqrt(sigz2))
        plt.title(ts)
        plt.xlim(self.PhyMin[adx], self.PhyMax[adx])
        plt.xlabel(f"{xname}")
        plt.ylabel(fn_name)
        plt.grid()
        plt.legend()
        # ax = plt.gca()
        # plt.ylim((min(y1),min(y1)+(max(y1)-min(y1))*1.5))
        # ins = ax.inset_axes([0.1,0.45,0.4,0.5])
        # ins.plot(x_small,y2,'g.-',markersize=7)
        # ins.plot(x, fn_x_val, 'ro')
        # # ins.plot(X, MLE_X, 'ko')
        # ins.grid()

