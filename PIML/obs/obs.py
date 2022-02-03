import logging
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
# from PIML.method.llh import LLH
from PIML.util.basespec import BaseSpec
from PIML.util.util import Util

class Obs(BaseSpec):
    def __init__(self):
        self.DATADIR = '/home/swei20/LV/data/fisher/'
        self.step = None
        self.sky_H = None
        self.sky_in_res = None
        
    def init_sky(self, wave_H, step, flux_in_res=None):
        self.prepare_sky(wave_H, step)
        if flux_in_res is not None:
            self.prepare_snr(flux_in_res)

    def prepare_sky(self, wave, step):
        self.sky_H = self.init_sky_grid(wave)
        self.step = step
        self.sky_in_res = BaseSpec.resampleFlux_i(self.sky_H, step)
    
    def prepare_snr(self, flux_in_res):
        self.noise_level_grid = [0,10,20,30,40,50,100,200]
        # self.snrList = [11,22,33,55,110]
        self.snrList = [10, 20, 30, 50]

        snr2nl = self.get_snr2nl_fn(flux_in_res)
        self.nlList = snr2nl(self.snrList)
        logging.info(f"nlList: {self.nlList}")

    @staticmethod
    def get_sky_interp_fn(skyOG):
        cs = np.cumsum(skyOG[:,1])
        f = sp.interpolate.interp1d(skyOG[:,0], cs, fill_value=0)
        return f

    def load_skyOG(self):
        skyOG = np.genfromtxt(self.DATADIR +'skybg_50_10.csv', delimiter=',')
        skyOG[:, 0] = 10 * skyOG[:, 0]
        return skyOG

    def init_sky_grid(self, wave_H):
        skyOG = self.load_skyOG()
        sky_fn = Obs.get_sky_interp_fn(skyOG)
        sky_grid = np.diff(sky_fn(wave_H))
        sky_grid = np.insert(sky_grid, 0, sky_fn(wave_H[0]))
        print("sky_H", sky_grid.shape)
        return sky_grid


    def get_sigma_in_res(self, flux_in_res, noise_level=1):
        var_in_res = Obs.get_var(flux_in_res, self.sky_in_res, step=self.step)
        sigma_in_res = np.sqrt(var_in_res)
        sigma = noise_level * sigma_in_res
        return sigma



    def add_obs_to_flux(self, flux_in_res, noise_level, step):
        # act on flux without taking log
        if step > 1: 
            sky_in_res = self.sky_in_res
        else:
            sky_in_res = self.sky_H
        
        var_in_res = Obs.get_var(flux_in_res, sky_in_res, step=step)
        noise      = Obs.get_noise(var_in_res)
        obsflux_in_res = flux_in_res + noise_level * noise
        obsvar_in_res = var_in_res * noise_level**2
        return obsflux_in_res, obsvar_in_res

    @staticmethod
    def get_obsflux_N(flux_in_res, var_in_res, noise_level, N):
        fluxs = np.tile(flux_in_res, (N, 1)) 
        var_in_res_N = np.tile(var_in_res, (N, 1)) 

        noise = Obs.get_noise(var_in_res_N)
        obsfluxs = fluxs + noise_level * noise
        return obsfluxs

    def add_obs_to_flux_N(self, flux_in_res, noise_level, step, N):
        var_in_res = Obs.get_var(flux_in_res, self.sky_in_res, step=step)
        print("noise_level", noise_level)
        obsvar_in_res = var_in_res * noise_level**2
        obsflux_in_res = Obs.get_obsflux_N(flux_in_res, var_in_res, noise_level, N)
        return obsflux_in_res, obsvar_in_res


    @staticmethod
    def get_avg_snr(fluxs, top=10):
        if isinstance(fluxs, list) or (len(fluxs.shape)>1):
            SNs = []
            for nsflux in fluxs[:top]:
                SNs.append(Util.get_snr(nsflux))
            return np.mean(SNs)
        else:
            print("not list")
            return Util.get_snr(fluxs)


    def get_snr2nl_fn(self, flux_in_res):
        #-----------------------------------------
        # choose the noise levels so that the S/N 
        # comes at around the predetermined levels
        #-----------------------------------------
        
        flux_shape = flux_in_res.shape[-1]
        if flux_shape == self.sky_in_res.shape[0]:
            step = self.step
            sky = self.sky_in_res
        elif flux_shape == self.sky_H.shape[0]:
            step = 0
            sky = self.sky_H
        else:
            raise ValueError("flux_in_res and sky_in_res should have the same shape")

        var_in_res = Obs.get_var(flux_in_res, sky, step=step)
        noise      = Obs.get_noise(var_in_res)

        SN = []
        for noise_level in self.noise_level_grid:
            ssobs = flux_in_res + noise_level * noise
            sn    = Util.get_snr(ssobs) / np.sqrt(2) # bosz R5000 = R10000, R = np.sqrt(10000/5000)
            if step > 1: sn = sn / np.sqrt(step) # getting snr at inst. resolution
            SN.append(sn)
        logging.info(f"snr2nl-SN: {SN}")
        f = sp.interpolate.interp1d(SN, self.noise_level_grid, fill_value=0)
        return f


    @staticmethod
    def get_var(ssm, skym, step=1):
        #--------------------------------------------
        # Get the total variance
        # BETA is the scaling for the sky
        # VREAD is the variance of the white noise
        # This variance is still scaled with an additional
        # factor when we simuate an observation.
        #--------------------------------------------
        assert ssm.shape[-1] == skym.shape[0]
        BETA  = 10.0
        VREAD = 16000
        varm  = ssm + BETA*skym + VREAD
        # return varm
        if step <= 1: 
            return varm
        else:
            return np.divide(varm, step)

    @staticmethod
    def get_noise(varm):
        noise = np.random.normal(0, np.sqrt(varm), np.shape(varm))
        return noise

#plot ---------------------------------------------------------------------------------
    @staticmethod
    def plot_noisy_spec(wave, flux_in_res, obsflux_in_res, pmt0):
        plt.figure(figsize=(9,3), facecolor='w')
        SN = Util.get_snr(obsflux_in_res)
        plt.plot(wave, obsflux_in_res, lw=1, label=f"SNR={SN:.1f}", color="gray")
        plt.plot(wave, flux_in_res, color="r")
        name = Util.get_pmt_name(*pmt0)
        plt.title(f"{name}")
        plt.legend()
        plt.xlabel("Wavelength [A]")
        plt.ylabel("Flux [erg/s/cm2/A]")



    