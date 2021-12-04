import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from PIML.method.llh import LLH
from PIML.util.basespec import BaseSpec

class Obs(BaseSpec):
    def __init__(self):
        self.DATADIR = '/home/swei20/LV/data/fisher/'
        self.sky = None
        self.noise_level_grid = [2,5,10,20,50,100,200,500]
        self.snrList = [11,22,33,55,110]
        self.nlList = None
        self.LLH = LLH()
        self.get_snr_from_nl = None



    def load_sky_H(self):
        sky = np.genfromtxt(self.DATADIR +'skybg_50_10.csv', delimiter=',')
        sky[:, 0] = 10 * sky[:, 0]
        return sky

    def prepare_sky(self, wave, flux_in_res, step):
        sky_H = self.load_sky_H()
        self.sky_in_res = Obs.resampleSky(sky_H, wave, step)
        self.get_snr_from_nl = self.interp_nl_fn(flux_in_res)
        self.nlList = self.get_snr_from_nl(self.snrList)

    def add_obs_to_flux(self, flux_in_res, noise_level):
        var_in_res = Obs.get_var(flux_in_res, self.sky_in_res)
        noise      = Obs.get_noise(var_in_res)
        obsflux_in_res = flux_in_res + noise_level * noise
        obsvar_in_res = var_in_res * noise_level**2
        return obsflux_in_res, obsvar_in_res


    @staticmethod
    def get_obsflux_N(flux_in_res, var_in_res, noise_level, N):
        fluxs = np.tile(flux_in_res, (N, 1)) 
        noise = Obs.get_noise_N(var_in_res, N)
        obsfluxs = fluxs + noise_level * noise
        return obsfluxs

    def add_obs_to_flux_N(self, flux_in_res, noise_level, N):
        var_in_res = Obs.get_var(flux_in_res, self.sky_in_res)
        obsvar_in_res = var_in_res * noise_level**2
        obsflux_in_res = Obs.get_obsflux_N(flux_in_res, var_in_res, noise_level, N)
        return obsflux_in_res, obsvar_in_res

#noise ---------------------------------------------------------------------------------
    @staticmethod
    def get_snr(flux):
        #--------------------------------------------------
        # estimate the S/N using Stoehr et al ADASS 2008
        #    signal = median(flux(i))
        #    noise = 1.482602 / sqrt(6.0) *
        #    median(abs(2 * flux(i) - flux(i-2) - flux(i+2)))
        #    DER_SNR = signal / noise
        #--------------------------------------------------
        s1 = np.median(flux)
        s2 = np.abs(2*flux-sp.ndimage.shift(flux,2)-sp.ndimage.shift(flux,-2))
        n1 = 1.482602/np.sqrt(6.0)*np.median(s2)
        sn = s1/n1
        return sn



    def interp_nl_fn(self, flux_in_res):
        #-----------------------------------------
        # choose the noise levels so that the S/N 
        # comes at around the predetermined levels
        #-----------------------------------------

        var_in_res = Obs.get_var(flux_in_res, self.sky_in_res)
        noise      = Obs.get_noise(var_in_res)

        SN = []
        for noise_level in self.noise_level_grid:
            ssobs = flux_in_res + noise_level * noise
            sn    = Obs.get_snr(ssobs)
            SN.append(sn)
        f = sp.interpolate.interp1d(SN, self.noise_level_grid, fill_value=0)
        return f



    def make_nlList(self, flux_H, skym, step=5):
        #-----------------------------------------
        # choose the noise levels so that the S/N 
        # comes at around the predetermined levels
        #-----------------------------------------
        
        if self.snrList is None:
            self.snrList = [11,22,33,55,110]
        if self.noise_level_grid is None:
            self.noise_level_grid = [2,5,10,20,50,100,200,500]



        # ssm   = Util.getModel(ss,0)
        ssm   = Obs.resampleFlux_i(flux_H, step)       
        varm  = Obs.get_var(ssm,skym)
        noise = Obs.get_noise(varm)  

        SN = []
        for noise_level in self.noise_level_grid:
            ssobs = ssm + noise_level * noise
            sn    = Obs.get_snr(ssobs)
            SN.append(sn)
        f = sp.interpolate.interp1d(SN, self.noise_level_grid, fill_value=0)
        
        noise_level_interpd = f(self.snrList)  
        return noise_level_interpd


    
    @staticmethod
    def get_var(ssm, skym):
        #--------------------------------------------
        # Get the total variance
        # BETA is the scaling for the sky
        # VREAD is the variance of the white noise
        # This variance is still scaled with an additional
        # factor when we simuate an observation.
        #--------------------------------------------
        BETA  = 10.0
        VREAD = 16000
        varm  = ssm + BETA*skym + VREAD
        return varm

    @staticmethod
    def get_noise_N(varm, N):
        out = np.zeros((N,varm.shape[0]))
        for i in range(N):
            out[i] = np.random.normal(0, np.sqrt(varm), len(varm))
        return out

    @staticmethod
    def get_noise(varm):
        #--------------------------------------------------------
        # given the noise variance, create a noise realization
        # using a Gaussian approximation
        # Input
        #  varm: the variance in m-pixel resolution
        # Output
        #  noise: nosie realization in m-pixels
        #--------------------------------------------------------
        # np.random.seed(42)
        noise = np.random.normal(0, np.sqrt(varm), len(varm))
        return noise

    # @staticmethod
    # def getObs(sconv,skym,rv, noise_level, step=5):
    #     #----------------------------------------------------
    #     # get a noisy spectrum for a simulated observation
    #     #----------------------------------------------------
    #     # inputs
    #     #   sconv: the rest-frame spectrum in h-pixels, convolved
    #     #   skym: the sky in m-pixels
    #     #   rv  : the radial velocity in km/s
    #     #   noise_level  : the noise amplitude
    #     # outputs
    #     #   ssm : the shifted, resampled sepectrum in m-pix
    #     #   varm: the variance in m-pixels
    #     #-----------------------------------------------
    #     # get shifted spec and the variance
    #     #-------------------------------------
    #     ssm   = Obs.getModel(sconv, rv, step=step)
    #     varm  = Obs.get_var(ssm,skym)
    #     noise = Obs.get_noise(varm)  
    #     #---------------------------------------
    #     # add the scaled noise to the spectrum
    #     #---------------------------------------
    #     ssm = ssm + noise_level * noise
    #     return ssm
    


    @staticmethod
    def get_random_grid_pmt(para, N_pmt):
        idx = np.random.randint(0, len(para), N_pmt)
        pmts = para[idx]
        return pmts


#plot ---------------------------------------------------------------------------------
    @staticmethod
    def plot_noisy_spec(wave, flux_in_res, obsflux_in_res, pmt0):
        plt.figure(figsize=(9,3), facecolor='w')
        SN = Obs.get_snr(obsflux_in_res)
        plt.plot(wave, obsflux_in_res, lw=1, label=f"SNR={SN:.1f}", color="gray")
        plt.plot(wave, flux_in_res, color="r")
        name = Obs.get_pmt_name(*pmt0)
        plt.title(f"{name}")
        plt.legend()
        plt.xlabel("Wavelength [A]")
        plt.ylabel("Flux [erg/s/cm2/A]")

    @staticmethod
    def plot_spec(wave, flux, pmt=None):
        plt.figure(figsize=(9,3), facecolor='w')
        plt.plot(wave, flux)
        if pmt is not None:
            name = Obs.get_pmt_name(*pmt)
            plt.title(f"{name}")
        plt.xlabel("Wavelength [A]")
        plt.ylabel("Flux [erg/s/cm2/A]")

    