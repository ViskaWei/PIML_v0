
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
        self.topk = None
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

    def init(self, W, R, Res, step, topk=10, onPCA=1):
        self.init_W(W)
        self.init_R(R)
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
        self.flux0 = self.get_model(self.PhyMid, onGrid=1, plot=1)
        self.init_sky(self.wave_H, self.flux0, step)

        self.logflux = self.Obs.safe_log(self.flux)
        self.interp_obs, self.interp_model = self.run_step_rbf(self.logflux, onPCA=onPCA)
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
            mu, pcflux = self.run_step_pca(logflux, top=self.topk)
            interp_obs_fn, interp_model_fn = self.build_PC_rbf(mu, pcflux)
            return interp_obs_fn, interp_model_fn
        else:
            interp_fn = self.build_logflux_rbf(logflux)
            return interp_fn, interp_fn

    def build_logflux_rbf(self, logflux):
        rbf_interp = self.RBF.train_rbf(self.pdx0, logflux)
        rbf = self.RBF.build_rbf(rbf_interp, self.pmt2pdx_scaler, np.exp)
        return rbf

    def build_PC_rbf(self, mu, pcflux):
        rbf_interp_mu = self.RBF.train_rbf(self.pdx0, mu)
        rbf_mu = self.RBF.build_rbf(rbf_interp_mu, self.pmt2pdx_scaler, None)
        self.rbf_mu = rbf_mu
        rbf_interp_coeff = self.RBF.train_rbf(self.pdx0, pcflux)
        rbf_coeff = self.RBF.build_rbf(rbf_interp_coeff, self.pmt2pdx_scaler, None)
        self.rbf_coeff = rbf_coeff
        
        def interp_obs_fn(x):
            mu = rbf_mu(x)
            coeff = rbf_coeff(x)
            logflux = coeff.dot(self.eigv) + mu
            return np.exp(logflux)

        def interp_model_fn(x):
            coeff = rbf_coeff(x)
            lognorm = coeff.dot(self.eigv)
            return np.exp(lognorm)

        return interp_obs_fn, interp_model_fn

    def run_step_pca(self, logflux, top=10):

        mu = np.mean(logflux, axis=1)
        normflux = logflux - mu[:, None]

        u,s,v = np.linalg.svd(normflux, full_matrices=False)
        
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
        assert (normflux.dot(v.T) - pcflux).max() < 1e-5
        return mu, pcflux

    def test_rbf(self, pmt1, pmt2, pmt=None):
        flux1, flux2 = self.get_model(pmt1,onGrid=1),  self.get_model(pmt2,onGrid=1)
        if pmt is None: pmt = 0.5 * (pmt1 + pmt2)
        interpFlux = self.interp_obs(pmt)
        plt.plot(self.wave, interpFlux, label= pmt)
        plt.plot(self.wave, flux1, label = pmt1)
        plt.plot(self.wave, flux2, label = pmt2)
        plt.legend()

# model------------------------------------------------------------------------
    def get_model(self, pmt, norm=0, onGrid=0, plot=0):
        if norm:
            flux = self.interp_model(pmt)
        else:
            if onGrid:
                fdx = self.Obs.get_fdx_from_pmt(pmt, self.para)
                flux = self.flux[fdx]
            else:
                flux = self.interp_obs(pmt)
        if plot: self.Obs.plot_spec(self.wave, flux, pmt=pmt)
        return flux
    

    def make_obs_from_pmt(self, pmt, noise_level=None, snr=None, N=1, plot=0, onPCA=0, onGrid=1):
        if noise_level is None:
            if snr == np.inf: 
                noise_level = 0
            else:
                noise_level = self.Obs.snr2nl(snr)
        flux = self.get_model(pmt, norm=0, onGrid=0)
        # np.random.seed(1015)
        obsfluxs, obsvar = self.Obs.add_obs_to_flux_N(flux, noise_level, self.step, N)
        if plot: self.Obs.plot_noisy_spec(self.wave, flux, obsfluxs[0], pmt)
        # if onPCA:
        #     normflux = self.get_model(pmt, norm=1)
        #     A = np.exp(self.rbf_mu(pmt))
        #     A0 = obsfluxs.mean(1).mean() / normflux.mean() 
        #     print(f"A = {A}, dA = {A-A0}")
        #     bias = self.get_bias(normflux, obsvar, A=A)
            
        #     # print(normflux - flux)
        #     if N == 1: obsfluxs = obsfluxs[0]
        #     return obsfluxs.mean(0), obsvar, bias, A
        else:
            if N == 1: obsfluxs = obsfluxs[0]
            return obsfluxs, obsvar

    def eval_pca_bias(self, pmt, N, noise_level=None, snr=None):
        if (noise_level is None) & (snr is None): 
            raise "noise level or snr not specified"
        obsfluxs, obsvar,= self.make_obs_from_pmt(pmt, noise_level=noise_level,snr=snr, N=N, onPCA=1, plot=0)
        bk = np.log(obsfluxs).dot(self.eigv.T).mean(0)
        ak = self.rbf_coeff(pmt)
        mp = self.get_model(pmt, norm=1)
        A = np.exp(self.rbf_mu(pmt))
        Amp = A * mp
        x =  obsvar ** 0.5 / Amp
        bias_all = self.eigv.dot(np.log(1 + x))
        bias_1st_order = self.eigv.dot(x)
        bias_2nd_order = 1/2 * self.eigv.dot(obsvar / Amp**2)
        
        return bk - ak, (bias_all, bias_1st_order, bias_2nd_order)

    def plot_theory_bias(self, bias, ax=None, title=None):
        if ax is None: 
            f, ax = plt.subplots(1, figsize=(6,4))
        b0, b1, b2 = bias
        ax.plot(b0, b0, 'ko', label="log(1+x)")
        ax.plot(b0, b1, 'rx', label = "x")
        ax.plot(b0, b2, 'bo', label = "$x^2$/2")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("log(1+x)")
        if title is not None: ax.set_title(title)
        ax.legend()
    
    def plot_exp_bias(self, ak, diffs, labels=None, ax=None, title=None):
        
        if ax is None: 
            f, ax = plt.subplots(1, figsize=(6,4))
        ak0 = abs(ak)
        for ii, diff in enumerate(diffs):
            d0 = abs(diff) 
            ax.plot(ak0, d0 / ak0, 'o', label=labels[ii])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("|$a_k$|")
        if title is not None: ax.set_title(title)
        ax.legend()





    def get_bias(self, flux, obsvar, A=1):
        x = np.divide(obsvar**0.5, A * flux)
        bias1 = self.eigv.dot(x)
        bias_all = self.eigv.dot(np.log(1+x))
        bias2 = 0.5 * self.eigv.dot(np.divide(obsvar, A**2 * flux**2))
        return bias1, bias2, bias_all
        
    def eval_coeff_bias(self, pmt, snr=10, N_obs=10, plot=0, N_plot=None):
        coeff = self.rbf_coeff(pmt)
        obsfluxs, obsvar, bias, A = self.make_obs_from_pmt(pmt, snr, N=N_obs, onPCA = self.onPCA)
        logobsflux = self.Obs.safe_log(obsfluxs)
        COEFF = logobsflux.dot(self.eigv.T)
        if snr !=np.inf: 
            snr = self.Obs.get_avg_snr(obsfluxs, top = N_obs)
        if plot: 
            if N_plot is None: N_plot = self.topk // 2 -1
            self.plot_eval_coeff_bias(COEFF, coeff, pmt, snr, N_plot = N_plot)
        print(f"diff = {COEFF.mean(0) - coeff - bias[1]}")
        print(f"diff_log(1+x) = {COEFF.mean(0) - coeff - bias[2]}")

        return COEFF, coeff, bias, snr, obsfluxs.mean(0), obsvar,A



#LLH --------------------------------------------------


    def eval_LLH_at_pmt(self, pmt, pdxs=[1], snr=10, N_obs=10, plot=0):
        obsfluxs , obsvar = self.make_obs_from_pmt(pmt, snr, N=N_obs)
        if snr !=np.inf: 
            snr = self.Obs.get_avg_snr(obsfluxs, top = N_obs)
        fns = self.LLH.get_eval_LLH_fns(pdxs, pmt, obsvar)
        preds = []
        for fn_pdx, fn in fns.items():
            pred_x = self.LLH.collect_estimation(fn, obsfluxs, fn_pdx, x0=self.PhyMid)
            preds.append(pred_x)
        preds = np.array(preds).T
        if plot:
            self.plot_eval_LLH(pmt, preds, pdxs, snr)
        return preds, snr


    def plot_eval_coeff_bias(self, pred, truth, pmt, snr, N_plot, n_box=0.5):
        fns = self.PLT.flow_fn_i(pred, truth, snr, legend=0)
        f = self.PLT.plotN(N_plot=N_plot, fns = fns, lbl="PC - a")
        name = self.Obs.get_pmt_name(*pmt)
        f.suptitle(f'{name} || snr {snr:.1f}')
        f.tight_layout()



    def plot_coeff_cdxs(self, COEFF, coeff, pmt, snr, N=3):
        N = np.min([N, len(coeff)])
        f, axs = plt.subplots(1,N, figsize=(N*3,3), facecolor="w")
        for ii in range(N):
            ax = axs[ii]
            self.plot_coeff_cdx(coeff, COEFF, 2*ii, 2*ii + 1, ax=ax)
        name = self.Obs.get_pmt_name(*pmt)
        f.suptitle(f'{name} || snr {snr:.1f}')
        f.tight_layout()

    def eval_coeff_snr(self, snr, pmts=None, N_pmt=10, N_obs=10, N_plot=None):
        if pmts is None: pmts = self.get_random_grid_pmt(N_pmt)
        if N_plot is None: N_plot = self.topk // 2 - 1
        fns = []
        SNRs= []
        for pmt in tqdm(pmts):
            COEFF, coeff, bias, SNR = self.eval_coeff_bias(pmt, snr=snr, N_obs=N_obs, plot=0, N_plot=N_plot)
            fns_pmt = self.PLT.flow_fn_i(COEFF, coeff, legend=0)
            fns = fns + fns_pmt
            SNRs.append(SNR)
        f = self.PLT.plotN(N_plot=N_plot, fns = fns, lbl="PC - a")
        name = self.Obs.get_pmt_name(*pmt)
        f.suptitle(f'{name} || snr {snr:.1f}')
        f.tight_layout()


    def eval_coeff(self, snrList=[50], pmts=None, N_pmt=10, N_obs=10, N_plot=None):
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
                COEFF, coeff, bias, SNR = self.eval_coeff_bias(pmt, snr=snr, N_obs=N_obs, plot=0, N_plot=N_plot)
                fns_pmt = self.PLT.flow_fn_i(COEFF, coeff, legend=0)
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

    def plot_coeff_cdx(self, coeff, COEFF, cdx1, cdx2, ax=None):
        if ax is None: fig, ax = plt.subplots(1,1)
        data1, data2 = COEFF[:,cdx1], COEFF[:,cdx2]
        ax.plot(data1, data2, "o", color="gray", markersize=2)
        ax.plot(coeff[cdx1], coeff[cdx2], "ro", label="$a_k$")
        ax.plot(data1.mean(), data2.mean(), "go", label="$b_k$")
        ax.legend()

        ax.set_xlabel(f"PC - a{cdx1}")
        ax.set_ylabel(f"PC - a{cdx2}") 


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

    def plot_coeff_on_pmt(self, fn, topk=10, axis="T", fn_name=""):
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

