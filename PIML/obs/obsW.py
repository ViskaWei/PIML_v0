import numpy as np
import scipy as sp
import logging
import matplotlib.pyplot as plt
from PIML.util.util import Util
from PIML.obs.baseobs import BaseObs

class ObsW(BaseObs):
    def __init__(self, W, step=None):
        self.DATADIR = '/home/swei20/LV/data/fisher/'
        self.W = W
        self.instrRes = Util.DWires[W[:1]]
        self.step = step or Util.DWstep[W]
        self.sky_H = None
        self.sky_in_res = None


    def init_W(self, wave_H, flux_in_res=None):
        self.sky_H, self.sky_in_res = self.set_sky(wave_H, self.step)
        if flux_in_res is not None:
            self.init_snr_W(flux_in_res)


    def init_snr_W(self, flux_in_res):
        self.noise_level_grid = [10, 30,40,50,100,200, 300, 400, 500]
        self.snrList = [10, 20, 30]

        self.snr2nl, self.nl2snr = self.get_snr2nl_fn(flux_in_res)
        self.nlList = self.snr2nl(self.snrList)
        logging.info(f"nlList: {self.nlList}")  
        
    def get_snr2nl_fn(self, flux_in_res):
        #-----------------------------------------
        # choose the noise levels so that the S/N 
        # comes at around the predetermined levels
        #-----------------------------------------
        sky, step = self.get_in_res_from_flux(flux_in_res)
        print(sky.shape, step)
        dotSqrt = 2 * 5000 /  self.instrRes
        logging.info(f"instrument Res = {self.instrRes},  dotSqrt of {dotSqrt:.2f}")
        f, f_inv = BaseObs._get_snr2nl_fn(flux_in_res, sky, step, self.noise_level_grid, dotSqrt, nAvg=1)
        return f, f_inv

    def make_obsflux(self, flux_in_res, noise_level=1):
        return BaseObs._make_obsflux(flux_in_res, self.sky_in_res, self.step, noise_level)

    def make_obsflux_sigma(self, flux_in_res, noise_level=1):
        return BaseObs._make_obsflux_sigma(flux_in_res, self.sky_in_res, self.step, noise_level)

    def get_in_res_from_flux(self, flux_in_res):
        flux_shape = flux_in_res.shape[-1]
        if flux_shape == self.sky_in_res.shape[0]:
            step = self.step
            sky = self.sky_in_res
        elif flux_shape == self.sky_H.shape[0]:
            step = 0
            sky = self.sky_H
        else:
            raise ValueError("flux_in_res and sky_in_res should have the same shape")
        return sky, step
        
    def make_obsflux_N(self, N, flux_in_res, noise_level):
        return BaseObs._make_obsflux_N(N, flux_in_res, self.sky_in_res, self.step, noise_level)

    def get_sigma_from_flux(self, flux_in_res, noise_level=1):
        var_in_res = BaseObs.get_var(flux_in_res, self.sky_in_res, step=self.step)
        sigma_in_res = np.sqrt(var_in_res)
        sigma = noise_level * sigma_in_res
        return sigma

