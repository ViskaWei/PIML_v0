import os
import h5py
import numpy as np
from PIML import obs
from PIML.util.basebox import BaseBox
from PIML.util.baseplot import BasePlot

from PIML.method.llh import LLH
from PIML.method.rbf import RBF
from PIML.method.bias import Bias
import matplotlib.pyplot as plt
from tqdm import tqdm

from PIML.util.util import Util

# class testBoxWR(BoxWR):


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
        self.LLH = LLH()
        self.topk = None



# init------------------------------------------------------------------------
    def init(self, W, R, Res, step, topk=10, onPCA=1):
        self.init_WR(W, R)
        self.init_plot_R()
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

        self.logflux = Util.safe_log(self.flux)

        self.interp_flux_fn, self.interp_model_fn, self.rbf_sigma, self.interp_bias_fn = self.run_step_rbf(self.logflux, onPCA=onPCA, Obs=self.Obs)
        self.init_LLH()


    # def init(self, W, R, Res, step, topk=10, onPCA=1):
    #     self.init_WR(W, R)
    #     self.init_plot_R()
    #     self.Res = Res
    #     self.step = step
    #     self.onPCA = onPCA
    #     self.topk = topk
    #     wave_H, flux_H, self.pdx, self.para = self.IO.load_bosz(Res, RR=self.RR)
    #     self.pdx0 = self.pdx - self.pdx[0]
    #     self.wave_H, flux_H = self.get_flux_in_Wrange(wave_H, flux_H)
    #     # self.flux_H = flux_H
    #     self.Mdx = self.Obs.get_fdx_from_pmt(self.PhyMid, self.para)
    #     self.fluxH0 = flux_H[self.Mdx]

    #     self.wave, self.flux = self.downsample(flux_H)
    #     self.Npix = len(self.wave)
    #     self.flux0 = self.get_model(self.PhyMid, onGrid=1, plot=1)

    #     self.init_sky(self.wave_H, self.flux0, step)

    #     self.logflux = Util.safe_log(self.flux)

    #     self.interp_flux_fn, self.interp_model_fn, self.rbf_sigma, self.interp_bias_fn = self.run_step_rbf(self.logflux, onPCA=onPCA, Obs=self.Obs)
    #     self.init_LLH()



 

    def init_LLH(self): 
        self.LLH.get_model = lambda x: self.get_model(x, onGrid=0, plot=0)
        self.LLH.x0 = self.PhyMid
        self.LLH.PhyMin = self.PhyMin
        self.LLH.PhyMax = self.PhyMax




    def get_random_grid_pmt(self, nPmt):
        pmts = Util.get_random_grid_pmt(self.para, nPmt)
        return pmts

    def get_random_pmt(self, nPmt, nPara=5, method="halton"):
        pmts = Util.get_random_uniform(nPmt, nPara, method=method, scaler=self.minmax_rescaler)
        return pmts



#RBF------------------------------------------------------------------------
    def run_step_rbf(self, logflux, onPCA=1, Obs=None):
        self.RBF = RBF(coord=self.pdx0, coord_scaler=self.pmt2pdx_scaler)
        if onPCA:
            logA, pcflux = self.run_step_pca(logflux, top=self.topk)
            interp_AModel_fn, interp_model_fn, self.rbf_logA, self.rbf_ak = self.RBF.build_PC_rbf_interp(logA, pcflux, self.eigv)
            
            if Obs is None:
                return interp_AModel_fn, interp_model_fn
            else:
                def rbf_sigma(pmt, noise_level, toPC=0):
                    AModel = interp_AModel_fn(pmt, log=0)
                    var_in_res = Obs.get_var(AModel, Obs.sky_in_res, step=self.step)
                    sigma_in_res = np.sqrt(var_in_res)
                    sigma_ak = np.divide(noise_level * sigma_in_res, AModel)
                    if toPC:
                        return sigma_ak.dot(self.eigv.T)
                    else:
                        return sigma_ak
            
                def interp_bias_fn(sigma_ak, X=None):
                    if X is None: X = np.random.normal(0,1, self.Npix)
                    bias = self.eigv.dot(np.multiply(X, sigma_ak))
                    return bias
                return interp_AModel_fn, interp_model_fn, rbf_sigma, interp_bias_fn
        else:
            interp_flux_fn = self.RBF.build_logflux_rbf_interp(logflux)
            if Obs is None:
                return interp_flux_fn, interp_flux_fn
            else:
                def rbf_sigma(pmt, noise_level):
                        AModel = interp_flux_fn(pmt, log=0)
                        var_in_res = Obs.get_var(AModel, Obs.sky_in_res, step=self.step)
                        sigma_in_res = np.sqrt(var_in_res)
                        stdmag = sigma_in_res * noise_level
                        return sigma_in_res
                
                def interp_bias_fn(stdmag, X=None):
                    if X is None: X = np.random.normal(0,1, self.Npix)
                    bias =np.multiply(X, stdmag)
                    return bias
                return rbf_sigma, interp_bias_fn, 

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
        interpFlux = self.interp_flux_fn(pmt)
        plt.plot(self.wave, interpFlux, label= pmt)
        plt.plot(self.wave, flux1, label = pmt1)
        plt.plot(self.wave, flux2, label = pmt2)
        plt.legend()

# model------------------------------------------------------------------------
    def get_model(self, pmt, norm=0, onGrid=0, plot=0):
        if norm:
            model= self.interp_model_fn(pmt)
            if plot: BasePlot.plot_spec(self.wave, model, pmt=pmt)
            return model
        else:
            if onGrid:
                fdx = self.Obs.get_fdx_from_pmt(pmt, self.para)
                Amodel = self.flux[fdx]
            else:
                Amodel = self.interp_flux_fn(pmt)
            if plot: BasePlot.plot_spec(self.wave, Amodel, pmt=pmt)
            return Amodel
    
    def get_bk_fn_from_pmt(self, pmt, noise_level):
        ak = self.rbf_ak(pmt)
        stdmag = self.rbf_sigma(pmt, noise_level)
        def add_bias(X=None):
            bias = self.interp_bias_fn(stdmag, X)
            return ak + bias
        return add_bias

    def get_bk_fns(self, noise_level, pmts=None, nPmts=1, out_bks=1):
        if pmts is None: 
            pmts = self.get_random_pmt(nPmts)
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
        if pmt is None: pmt = self.get_random_pmt(1)[0]
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

    def estimate_snr(self, NL):
        obsfluxH0, _ = self.Obs.add_obs_to_flux(self.fluxH0, NL, step=0)
        bosz_5000_snr_factor = np.sqrt(2)
        snr = Util.get_snr(obsfluxH0) / bosz_5000_snr_factor
        return snr

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
        AModel = self.interp_flux_fn(pmt, log=0)
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

    def eval_LLH_at_pmt(self, pmt, odxs=[1], noise_level=100, nObs=10, plot=0):
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
            snr = self.estimate_snr(noise_level)
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


    def eval_LLH_NL(self, noise_level=None, pmts=None, pdxs=[0,1,2], nPmt=10, nObs=10, n_box=0.5):
        if pmts is None: pmts = self.get_random_pmt(nPmt)
        fns = []
        for pmt in tqdm(pmts):
            preds_pmt = self.eval_LLH_at_pmt(pmt, pdxs, noise_level=noise_level, nObs=nObs, plot=0)
            fns_pmt = self.PLT.flow_fn_i(preds_pmt, pmt[pdxs], legend=0)
            fns = fns + fns_pmt

        f = self.PLT.plot_box(pdxs, fns = fns, n_box=n_box)
        snr = self.estimate_snr(noise_level)
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
    def prepare_trainset(self, N, pmts=None, noise_level=1, add_noise=False):
        if pmts is None: 
            pmts = self.get_random_pmt(N, method="random")
        else:
            pmts = pmts[:N]
        aks = self.rbf_ak(pmts)

        if add_noise:
            if noise_level <=1:
                return aks, pmts
            else:
                sigma = self.rbf_sigma(pmts, noise_level) 
                noiseMat = np.random.normal(0, sigma, sigma.shape)
                noisePC = noiseMat.dot(self.eigv.T) # convert noise into topk PC basis
                bks = aks + noisePC  
                return bks, pmts
        else:
            if noise_level <=1:
                return [aks, np.zeros_like(aks)], pmts
            else:
                sigma = self.rbf_sigma(pmts, 1) #noise_level =1 for now, will be add dynamically to NN later   
                return [aks, sigma], pmts

    def prepare_testset(self, N, pmts=None, noise_level=1, seed=None):
        if pmts is None: 
            pmts = self.get_random_pmt(N, method="random")
        else:
            pmts = pmts[:N]
        aks = self.rbf_ak(pmts)
        if noise_level <=1:
            return aks, pmts
        else:
            sigma = self.rbf_sigma(pmts, noise_level)
            if seed is not None: np.random.seed(seed)
            noiseMat = np.random.normal(0, sigma, sigma.shape)
            noisePC = noiseMat.dot(self.eigv.T) # convert noise into topk PC basis
            bks = aks + noisePC
            return bks, pmts

    def prepare_noiseset(self, pmt, noise_level, nObs):
        ak0 = self.rbf_ak(pmt)
        ak = np.tile(ak0, (nObs, 1))
        if noise_level <=1: 
            return ak
        else:
            sigma0 = self.rbf_sigma(pmt, noise_level)
            sigma = np.tile(sigma0, (nObs, 1))
            noiseMat = np.random.normal(0, sigma, sigma.shape)
            noisePC = noiseMat.dot(self.eigv.T)
            return ak + noisePC