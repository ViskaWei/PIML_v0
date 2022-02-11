import logging
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from PIML.util.basespec import BaseSpec
from PIML.util.util import Util

class BaseObs(BaseSpec):
    def __init__(self):
        self.DATADIR = '/home/swei20/LV/data/fisher/'
    


    def set_sky(self, wave, step):
        sky_H = self.init_sky_grid(wave)
        sky_in_res = BaseSpec.resampleFlux_i(sky_H, step)
        return sky_H, sky_in_res
    
    def init_sky_grid(self, wave_H):
        skyOG = self.load_skyOG()
        sky_fn = BaseObs.get_sky_interp_fn(skyOG)
        sky_grid = np.diff(sky_fn(wave_H))
        sky_grid = np.insert(sky_grid, 0, sky_fn(wave_H[0]))
        print("sky_H", sky_grid.shape)
        return sky_grid
    
    def load_skyOG(self):
        skyOG = np.genfromtxt(self.DATADIR +'skybg_50_10.csv', delimiter=',')
        skyOG[:, 0] = 10 * skyOG[:, 0]
        return skyOG

    @staticmethod
    def get_sky_interp_fn(skyOG):
        cs = np.cumsum(skyOG[:,1])
        f = sp.interpolate.interp1d(skyOG[:,0], cs, fill_value=0)
        return f
    
    @staticmethod
    def _get_snr2nl_fn(flux_in_res, sky_in_res, step, noise_level_grid, factor, nAvg=1):
        var_in_res = BaseObs.get_var(flux_in_res, sky_in_res, step=step)
        sigma_in_res = np.sqrt(var_in_res)
        noise = BaseObs.get_sigma_noise(sigma_in_res)
        SN = []
        for noise_level in noise_level_grid:
            obsfluxs = flux_in_res + noise_level * noise
            sn = Util.get_snr(obsfluxs, sigma_in_res, noise_level)
            sn = sn * np.sqrt(factor)
            SN.append(sn)
        logging.info(f"snr2nl-SN: {SN}")
        f = sp.interpolate.interp1d(SN, noise_level_grid, fill_value=0)
        return f


    @staticmethod
    def get_avg_snr(fluxs, sigma, top=10):
        if isinstance(fluxs, list) or (len(fluxs.shape)>1):
            SNs = []
            for nsflux in fluxs[:top]:
                SNs.append(Util.get_snr(nsflux, sigma))
            return np.mean(SNs)
        else:
            return Util.get_snr(fluxs, sigma)
    
    @staticmethod
    def _make_obsflux(flux_in_res, sky_in_res, step, noise_level):
        var_in_res = BaseObs.get_var(flux_in_res, sky_in_res, step=step)
        noise      = BaseObs.get_noise(var_in_res)
        obsflux_in_res = flux_in_res + noise_level * noise
        obsvar_in_res = var_in_res * noise_level**2
        return obsflux_in_res, obsvar_in_res

    @staticmethod
    def _make_obsflux_sigma(flux_in_res, sky_in_res, step, noise_level):
        var_in_res = BaseObs.get_var(flux_in_res, sky_in_res, step=step)
        sigma_in_res = np.sqrt(var_in_res)
        obssigma_in_res = sigma_in_res * noise_level
        noise      = BaseObs.get_sigma_noise(obssigma_in_res)
        obsflux_in_res = flux_in_res + noise
        return obsflux_in_res, obssigma_in_res


    @staticmethod
    def _make_obsflux_from_var_N(N, flux_in_res, var_in_res, noise_level):
        fluxs = np.tile(flux_in_res, (N, 1)) 
        var_in_res_N = np.tile(var_in_res, (N, 1)) 
        noise = BaseObs.get_noise(var_in_res_N)
        obsfluxs = fluxs + noise_level * noise
        return obsfluxs
    
    @staticmethod
    def _make_obsflux_N(N, flux_in_res, sky_in_res, noise_level, step):
        var_in_res = BaseObs.get_var(flux_in_res, sky_in_res, step=step)
        print("noise_level", noise_level)
        obsvar_in_res = var_in_res * noise_level**2
        obsflux_in_res = BaseObs._make_obsflux_from_var_N(N, flux_in_res, var_in_res, noise_level)
        return obsflux_in_res, obsvar_in_res


    @staticmethod
    def get_var(flux, skym, step=1):
        #--------------------------------------------
        # Get the total variance
        # BETA is the scaling for the sky
        # VREAD is the variance of the white noise
        # This variance is still scaled with an additional
        # factor when we simuate an observation.
        #--------------------------------------------
        assert flux.shape[-1] == skym.shape[0]
        BETA  = 10.0
        VREAD = 16000
        varm  = flux + BETA*skym + VREAD
        # return varm
        if step <= 1: 
            return varm
        else:
            return np.divide(varm, step)

    @staticmethod
    def get_noise(varm):
        noise = np.random.normal(0, np.sqrt(varm), np.shape(varm))
        return noise

    @staticmethod
    def get_sigma_noise(sigma):
        return np.random.normal(0, sigma, np.shape(sigma))

#plot ---------------------------------------------------------------------------------
    @staticmethod
    def plot_noisy_spec(wave, flux_in_res, obsflux_in_res, sigma, pmt0):
        plt.figure(figsize=(9,3), facecolor='w')
        if sigma is not None:
            SN = Util.get_snr(obsflux_in_res)
        plt.plot(wave, obsflux_in_res, lw=1, label=f"SNR={SN:.1f}", color="gray")
        plt.plot(wave, flux_in_res, color="r")
        name = Util.get_pmt_name(*pmt0)
        plt.title(f"{name}")
        plt.legend()
        plt.xlabel("Wavelength [A]")
        plt.ylabel("Flux [erg/s/cm2/A]")



    