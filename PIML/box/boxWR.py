import os
import h5py
import numpy as np
import logging
from tqdm import tqdm

from PIML.util.basebox import BaseBox
from PIML.util.baseplot import BasePlot

from PIML.method.llh import LLH
from PIML.method.rbf import RBF
from PIML.method.bias import Bias
import matplotlib.pyplot as plt

from PIML.util.util import Util



class BoxWR(BaseBox):
    def __init__(self):
        super().__init__()


        self.LLH = LLH()
        self.topk = None
        self.Res = None
        
    

# init------------------------------------------------------------------------
    def init(self, W, R, Res, step, topk=10, onPCA=1, fast=False):
        self.init_W(W)
        self.init_R(R)
        self.init_plot_R()
        self.Res = Res
        self.onPCA = onPCA
        self.topk = topk

        self.flux, self.pdx0, self.para, self.Obs = self.prepare_data_WR(W,R,Res,step,store=True)
        self.run_step_rbf(fast=fast)
        self.test_rbf(self.PhyMid, axis=1)

    def init_LLH(self): 
        self.LLH.get_model = lambda x: self.get_model(x, onGrid=0, plot=0)
        self.LLH.x0 = self.PhyMid
        self.LLH.PhyMin = self.PhyMin
        self.LLH.PhyMax = self.PhyMax

    def get_random_grid_pmt(self, nPmt):
        pmts = Util.get_random_grid_pmt(self.para, nPmt)
        return pmts



#RBF------------------------------------------------------------------------
    def run_step_rbf(self, fast=False):
        if self.onPCA:
            self.eigv0, self.pcflux, fns = self.prepare_PC_rbf(self.pdx0, self.pmt2pdx_scaler, self.flux, Obs=self.Obs, topk=self.topk, fast=fast)
            self.eigv = self.eigv0[:self.topk]
            self.rbf_flux, self.rbf_flux_sigma, self.rbf_ak, self.rbf_ak_sigma, self.rbf_logflux_sigma, self.rbf_logA, self.gen_nObs_noise = fns
        else:
            self.rbf_flux, self.rbf_flux_sigma = self.prepare_logflux_rbf(self.pdx0, self.pmt2pdx_scaler, self.flux, Obs=self.Obs)

    def test_rbf(self, pmt1, pmt2=None, axis=1, pmt=None):
        if pmt2 is None:
            pmt2 = np.copy(pmt1)
            pmt2[axis] += BaseBox.PhyTick[axis]
        flux1, flux2 = self.get_model(pmt1,onGrid=1),  self.get_model(pmt2,onGrid=1)
        if pmt is None: pmt = 0.5 * (pmt1 + pmt2)
        interpFlux = self.rbf_flux(pmt, log=0)
        plt.plot(self.wave, interpFlux, label= pmt)
        plt.plot(self.wave, flux1, label = pmt1)
        plt.plot(self.wave, flux2, label = pmt2)
        plt.legend()

# model------------------------------------------------------------------------
    def get_model(self, pmt, norm=0, onGrid=0, plot=0):
        if norm:
            model= self.rbf_flux(pmt, log=0, dotA=0)
            if plot: BasePlot.plot_spec(self.wave, model, pmt=pmt)
            return model
        else:
            if onGrid:
                fdx = Util.get_fdx_from_pmt(pmt, self.para)
                Amodel = self.flux[fdx]
            else:
                Amodel = self.rbf_flux(pmt, log=0, dotA=1)
            if plot: BasePlot.plot_spec(self.wave, Amodel, pmt=pmt)
            return Amodel
    
    def get_bk_fn_from_pmt(self, pmt, noise_level):
        ak = self.rbf_ak(pmt)
        stdmag = self.rbf_sigma(pmt, noise_level)
        def add_bias(X=None):
            #FIXME 
            bias = self.interp_bias_fn(stdmag, X)
            return ak + bias
        return add_bias

    def get_bk_fns(self, noise_level, pmts=None, nPmts=1, out_bks=1):
        if pmts is None: 
            pmts = self.get_random_pmts(nPmts)
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

    def get_bks_nObs_from_pmt(self, pmt=None, noise_level=1, nObs=1):
        if pmt is None: pmt = self.get_random_pmts(1)[0]
        bk_fn = self.get_bk_fn_from_pmt(pmt, noise_level)
        bk_nObs = np.zeros((nObs, self.topk))
        for ii in range(nObs):
            bk_nObs[ii] = bk_fn()
        return bk_nObs
        


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
        if N == 1: obsfluxs = obsfluxs[0]
        return obsfluxs, obsvar


#Bias------------------------------------------------------------------------

    def eval_pca_bias(self, pmt, N, noise_level=None, snr=None):
        self.Bias = Bias(self.eigv)
        if (noise_level is None):
            if (snr is None): 
                raise "noise level or snr not specified"
            else:
                noise_level= self.Obs.snr2nl(snr)
        obsfluxs, obsvar = self.make_obs_from_pmt(pmt, noise_level=noise_level,snr=snr, N=N, onPCA=1, plot=0)
        obssigma = np.sqrt(obsvar)
        AModel = self.rbf_flux(pmt, log=0, dotA=1)
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
        stdmag = self.rbf_sigma(pmt, noise_level)
        varmag = X20 * stdmag**2
        bias  = self.Bias.get_bias_from_stdmag(stdmag)
        biasX = self.Bias.get_bias_from_stdmag(np.multiply(X0, stdmag),  varmag)
        return bk - ak, bias, biasX

    def plot_pca_bias(self,  diffs, bias, biasX, pmt=None, diff_labels=None, N=None):
        if pmt is None: pmt = self.PhyMid
        title = self.RR + " " + Util.get_pmt_name(*pmt)
        f, axs = plt.subplots(1, 3, figsize=(16,4), facecolor="w")
        ak = self.rbf_ak(pmt)
        _=self.Bias.plot_theory_bias(ak, bias, ax=axs[0], N=N, title=title)
        if biasX is not None:
            _=self.Bias.plot_theory_bias(ak, biasX, ax=axs[1], log=1, theory=0, N=N, title=title)
        _=self.Bias.plot_exp_bias(ak, diffs, labels=diff_labels, ax=axs[2])

#LLH --------------------------------------------------

    def eval_LLH_at_pmt(self, pmt, odxs=[1], snr=10, nObs=10, plot=0):
        noise_level = self.Obs.snr2nl(snr)
        obsfluxs , obsvar = self.make_obs_from_pmt(pmt, noise_level=noise_level, N=nObs)
        # if snr !=np.inf: 
        #     snr = self.Obs.get_avg_snr(obsfluxs, top = nObs)
        fns = self.LLH.get_eval_LLH_fns(odxs, pmt, obsvar)
        preds = []
        for fn_pdx, fn in fns.items():
            pred_x = self.LLH.collect_estimation(fn, obsfluxs, fn_pdx, x0=self.PhyMid)
            preds.append(pred_x)
        preds = np.array(preds).T
        if plot:
            self.plot_eval_LLH(pmt, preds, odxs, snr)
        return preds


    def plot_eval_ak_bias(self, pred, truth, pmt, snr, N_plot, n_box=0.5):
        fns = self.PLT.flow_fn_i(pred, truth, snr, legend=0)
        f = self.PLT.plotN(N_plot=N_plot, fns = fns, lbl="PC - a")
        name = Util.get_pmt_name(*pmt)
        f.suptitle(f'{name} || snr {snr:.1f}')
        f.tight_layout()



    def plot_ak_cdxs(self, AK, ak, pmt, snr, N=3):
        N = np.min([N, len(ak)])
        f, axs = plt.subplots(1,N, figsize=(N*3,3), facecolor="w")
        for ii in range(N):
            ax = axs[ii]
            self.plot_ak_cdx(ak, AK, 2*ii, 2*ii + 1, ax=ax)
        name = Util.get_pmt_name(*pmt)
        f.suptitle(f'{name} || snr {snr:.1f}')
        f.tight_layout()

    def eval_ak_snr(self, snr, pmts=None, nPmt=10, nObs=10, N_plot=None):
        if pmts is None: pmts = self.get_random_grid_pmt(nPmt)
        if N_plot is None: N_plot = self.topk // 2 - 1
        fns = []
        SNRs= []
        for pmt in tqdm(pmts):
            AK, ak, bias, SNR = self.eval_ak_bias(pmt, snr=snr, nObs=nObs, plot=0, N_plot=N_plot)
            fns_pmt = self.PLT.flow_fn_i(AK, ak, legend=0)
            fns = fns + fns_pmt
            SNRs.append(SNR)
        f = self.PLT.plotN(N_plot=N_plot, fns = fns, lbl="PC - a")
        name = Util.get_pmt_name(*pmt)
        f.suptitle(f'{name} || snr {snr:.1f}')
        f.tight_layout()


    def eval_ak(self, snrList=[50], pmts=None, nPmt=10, nObs=10, N_plot=None):
        if pmts is None: pmts = self.get_random_grid_pmt(nPmt)
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
                AK, ak, bias, SNR = self.eval_ak_bias(pmt, snr=snr, nObs=nObs, plot=0, N_plot=N_plot)
                fns_pmt = self.PLT.flow_fn_i(AK, ak, legend=0)
                fns_snr = fns_snr + fns_pmt
                SNRs_snr.append(SNR)
            fns[snr] = fns_snr
            SNR_snr = np.mean(SNRs_snr)
            SNRs[snr] = SNR_snr
            self.PLT.plotN(N_plot=N_plot, fns = fns_snr, lbl="PC - a", axs=axs)
            axs[0].set_title(f'SNR {SNR_snr:.1f}')
        name = Util.get_pmt_name(*pmt)
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


    def eval_LLH_NL(self, snr, pmts=None, pdxs=[0,1,2], nPmt=10, nObs=10, n_box=0.5):
        if pmts is None: pmts = self.get_random_pmts(nPmt)
        noise_level = self.Obs.snr2nl(snr)
        fns = []
        for pmt in tqdm(pmts):
            preds_pmt = self.eval_LLH_at_pmt(pmt, pdxs, noise_level=noise_level, nObs=nObs, plot=0)
            fns_pmt = self.PLT.flow_fn_i(preds_pmt, pmt[pdxs], legend=0)
            fns = fns + fns_pmt

        f = self.PLT.plot_box(pdxs, fns = fns, n_box=n_box)
        f.suptitle(f"SNR={snr:.2f}")

    def eval_LLH(self, pmts=None, pdxs=[0,1,2], nPmt=10, n_box=0.5, snrList=None):
        if snrList is None: snrList = self.Obs.snrList
        # if pmts is None: pmts = self.get_random_grid_pmt(nPmt)
        for snr in snrList:
            self.eval_LLH_snr(snr, pmts, pdxs, nPmt, n_box)

    def plot_eval_LLH(self, pmt, pred, pdxs, snr, n_box=0.5):
        fns = self.PLT.flow_fn_i(pred, pmt[pdxs], snr, legend=0)
        f = self.PLT.plot_box(pdxs, fns = fns, n_box=n_box)
        f.suptitle(f"SNR = {snr}")



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


#DNN-----------------------------------------------------------------------------------------------------------------------
    

    def prepare_pmts(self, N, pmts=None, odx=None):
        if pmts is None: 
            rands, pmts = self.get_random_pmts(N, method="random", outRnd=True)
        else:
            pmts = pmts[:N]
            rands = self.minmax_scaler(pmts)        
        if odx is not None: rands = rands[:,odx]
        return rands, pmts
    
    def prepare_trainset(self, N, pmts=None, noise_level=1, add_noise=False, odx=None, topk=None):
        logging.info(f"Prepareing #{N} NL={noise_level} trainset")
        rands, pmts = self.prepare_pmts(N, pmts=pmts, odx=odx)
        if self.onPCA:
            out = self.prepare_trainset_onPCA_from_pmts(pmts,topk, noise_level=noise_level, add_noise=add_noise, seed=None)
        else:
            out = self.prepare_trainset_onFlux_from_pmts(pmts, noise_level=noise_level, add_noise=add_noise, seed=None)
        return out, rands, pmts

    def prepare_trainset_onPCA_from_pmts(self, pmts, topk, noise_level=1, add_noise=False, seed=None):
        if noise_level <1:
            aks = self.rbf_ak(pmts, topk=topk)
            out = aks if add_noise else [aks, np.zeros_like(aks)]
        else:
            if add_noise:
                out = self.aug_ak_in_pmts(pmts, noise_level, topk=topk, seed=seed)
            else:
                out = self.rbf_ak_sigma(pmts, topk=topk)
        return out

    def prepare_trainset_onFlux_from_pmts(self, pmts, noise_level=1, add_noise=False, seed = None):
        if noise_level < 1:
            out = self.rbf_flux(pmts, log=1, dotA=1)
            out = out if add_noise else [out, np.zeros_like(out)]
        else:
            if add_noise:
                out = self.aug_flux_in_pmts(pmts, noise_level, seed=seed)
            else:
                out = self.rbf_flux_sigma(pmts)
        return out

    def aug_ak_in_pmts(self, pmts, noise_level, topk=None, seed=None):
        aks, sigma = self.rbf_ak_sigma(pmts, topk=topk)
        if seed is not None: np.random.seed(seed)
        sigma = sigma * noise_level
        noiseMat = np.random.normal(0, sigma, sigma.shape)
        eigv = self.eigv0[:topk]
        noisePC = noiseMat.dot(eigv.T) # convert noise into topk PC basis
        bks = aks + noisePC
        return bks

    def aug_flux_in_pmts(self, pmts, noise_level, seed=None):
        logModel, sigma_log = self.rbf_flux_sigma(pmts)
        if seed is not None: np.random.seed(seed)
        sigma = sigma_log * noise_level
        logNoise = np.random.normal(0, sigma, sigma.shape)
        logObsfluxs = logModel + logNoise
        return logObsfluxs

    def prepare_testset(self, N, pmts=None, noise_level=1, topk=None, seed=None):
        pmts = self.get_random_pmts(N, method="random") if pmts is None else pmts[:N]
        if noise_level <1: 
            out = self.rbf_ak(pmts, topk=topk)
        else:
            out = self.aug_ak_in_pmts(pmts, noise_level, topk=topk, seed=seed)
        return out, pmts



    def prepare_noiseset(self, pmt, noise_level, nObs, topk=None):
        assert (noise_level > 1)
        out, sigma_log = self.rbf_ak_sigma(pmt, topk=topk) if self.onPCA else self.rbf_flux_sigma(pmt)
        obssigma = sigma_log * noise_level
        noise = self.gen_nObs_noise(obssigma, nObs, topk=topk)
        return out + noise