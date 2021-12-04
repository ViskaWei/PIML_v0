import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from PIML.util.util import Util




class LLH(Util):
    def __init__(self):
        super().__init__()
        self.PhyMin = None
        self.PhyMax = None
        self.x0 = None
    # likelihood---------------------------------------------------------------------------------

    def init_LLH(self, PhyMin, PhyMax, PhyMid):
        self.PhyMin = PhyMin
        self.PhyMax = PhyMax
        self.x0 = PhyMid

    @staticmethod
    def getLogLik(model, obsflux, var, nu_only=True):
        phi = np.sum(np.divide(np.multiply(obsflux, model), var))
        chi = np.sum(np.divide(np.multiply(model  , model), var))
        nu  = phi / np.sqrt(chi)    
        if nu_only: 
            return -nu
        else:
            return nu, phi, chi

    @staticmethod
    def lorentz(x, a,b,c,d):
        return a/(1+(x-b)**2/c**2) + d

    @staticmethod
    def estimate(fn, x0=None, bnds=None):
        if x0 is None: x0 = LLH.guessEstimation(fn)
        # print(f"x0 = {x0}")
        # print(f"bnds = {bnds}")
        out = sp.optimize.minimize(fn, x0, bounds = bnds, method="Nelder-Mead")
        if (out.success==True):
            X = out.x[0]
        else:
            X = np.nan
        return X

    @staticmethod
    def guessEstimation(fn):
        pass

    @staticmethod
    def collect_estimation(fn, obsfluxs, adx, x0=None):
        x0_adx = None if x0 is None else x0[adx] 
        Xs = []
        for obsflux in obsfluxs:
            fn_i = lambda x: fn(x, obsflux)
            X = LLH.estimate(fn_i, x0=x0_adx, bnds=None)
            Xs.append(X)
        return Xs


    def get_model(self):
        pass
        # raise("Not Implemented")

    def get_eval_LLH_fn(self, pdx, temp_pmt, obsvar):
        pmt = np.copy(temp_pmt)
        def fn(x, obsflux=None):
            pmt[pdx] = x
            tempflux_in_res = self.get_model(pmt)
            return LLH.getLogLik(tempflux_in_res, obsflux, obsvar, nu_only=True) 
        return fn

    def get_eval_LLH_fns(self, pdxs, pmt, obsvar):
        fns = {}
        for pdx in pdxs:
            fn = self.get_eval_LLH_fn(pdx, pmt, obsvar)
            fns[pdx] = fn
        return fns

    def getLogLik_pmt(self, temp_pmt, obsflux, obsvar, sky_mask0=None, nu_only=True):
        tempflux_in_res = self.get_model(temp_pmt)
        return LLH.getLogLik(tempflux_in_res, obsflux, obsvar, nu_only=nu_only)   
        
    def get_LLH_fn(self, pdx, temp_pmt, obsflux, obsvar, sky_mask0=None):
        pmt = np.copy(temp_pmt)
        def fn(x, nu_only=True):
            pmt[pdx] = x
            return self.getLogLik_pmt(pmt, obsflux, obsvar, 
                                    sky_mask0=sky_mask0, nu_only=nu_only)
        return fn



    def eval_pmt_on_axis(self, temp_pmt, x, obsflux, obsvar, axis="T", sky_mask0=None, plot=1):
        pdx = Util.PhyShort.index(axis)
        name = Util.get_pmt_name(*temp_pmt)
        print(f"Fitting with Template {name}")
        fn = self.get_LLH_fn(pdx, temp_pmt, obsflux, obsvar, sky_mask0=sky_mask0)
        self.fn = fn
        X = LLH.estimate(fn, x0=self.x0[pdx], bnds=None)
        print("estimate", X)
        if plot: 
            SN = Util.get_snr(obsflux)
            sigz2 = 0
            self.plot_pmt_on_axis(fn, x, X, SN, sigz2, pdx)
        return X

    def plot_pmt_on_axis(self, fn, x, X, SN, sigz2, pdx):
        x_large = np.linspace(self.PhyMin[pdx], self.PhyMax[pdx], 101)
        x_small = np.linspace(x - Util.PhyTick[pdx], x + Util.PhyTick[pdx], 25)
        y1 = []
        y2 = []
        for xi in x_large:
            y1.append(-1 * fn(xi))
        for xj in x_small:
            y2.append(-1 * fn(xj))

        MLE_x = -1 * fn(x)
        MLE_X = -1 * fn(X)
        plt.figure(figsize=(15,6), facecolor="w")
        plt.plot(x_large, y1,'g.-',markersize=7, label = f"llh")    
        plt.plot(x, MLE_x, 'ro', label=f"Truth {MLE_x:.2f}")
        plt.plot(X, MLE_X, 'ko', label=f"Estimate {MLE_X:.2f}")
        xname = Util.PhyLong[pdx]
        ts = f'{xname} Truth={x:.2f}K, {xname} Estimate={X:.2f}K, S/N={SN:3.1f}'
        # ts = ts +  'sigz={:6.4f} km/s,  '.format(np.sqrt(sigz2))
        plt.title(ts)
        plt.xlabel(f"{xname}")
        plt.ylabel("Log likelihood")
        plt.grid()
        plt.ylim((min(y1),min(y1)+(max(y1)-min(y1))*1.5))
        plt.legend()
        ax = plt.gca()
        ins = ax.inset_axes([0.1,0.45,0.4,0.5])
        ins.plot(x_small,y2,'g.-',markersize=7)
        ins.plot(x, MLE_x, 'ro')
        ins.plot(X, MLE_X, 'ko')
        ins.grid()