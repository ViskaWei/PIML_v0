import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from PIML.method.llh import LLH
from PIML.util.basespec import BaseSpec

class Obs(BaseSpec):
    def __init__(self):
        self.DATADIR = '/home/swei20/LV/data/fisher/'
        self.sky_fn = None
        self.skyOG = None
        self.sky_H = None
        self.sky_in_res = None
        self.noise_level_grid = [2,5,10,20,30,40,50,100,200,500,800]
        # self.snrList = [11,22,33,55,110]
        self.snrList = [10, 20, 30, 50]
        self.nlList = None
        self.LLH = LLH()
        self.snr2nl = None

    def get_sky_interp_fn(self):
        if self.skyOG is None: self.skyOG = self.load_sky_H()
        cs = np.cumsum(self.skyOG[:,1])
        f = sp.interpolate.interp1d(self.skyOG[:,0], cs, fill_value=0)
        return f

    def load_skyOG(self):
        skyOG = np.genfromtxt(self.DATADIR +'skybg_50_10.csv', delimiter=',')
        skyOG[:, 0] = 10 * skyOG[:, 0]
        return skyOG

    def init_sky_grid(self, wave_H):
        self.skyOG = self.load_skyOG()
        self.sky_fn = self.get_sky_interp_fn()
        sky_grid = np.diff(self.sky_fn(wave_H))
        sky_grid = np.insert(sky_grid, 0, self.sky_fn(wave_H[0]))
        print("sky_H", sky_grid.shape)
        return sky_grid

    def prepare_sky(self, wave, flux_in_res, step):
        self.sky_H = self.init_sky_grid(wave)
        self.step = step
        self.sky_in_res = Obs.resampleFlux_i(self.sky_H, step)
        self.snr2nl = self.get_snr2nl_fn(flux_in_res, step)
        self.nlList = self.snr2nl(self.snrList)



    def add_obs_to_flux(self, flux_in_res, noise_level, step):
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
        noise = Obs.get_noise_N(var_in_res, N)
        obsfluxs = fluxs + noise_level * noise
        return obsfluxs

    def add_obs_to_flux_N(self, flux_in_res, noise_level, step, N):
        var_in_res = Obs.get_var(flux_in_res, self.sky_in_res, step=step)
        print("noise_level", noise_level)
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

    @staticmethod
    def get_avg_snr(fluxs, top=10):
        if isinstance(fluxs, list) or (len(fluxs.shape)>1):
            SNs = []
            for nsflux in fluxs[:top]:
                SNs.append(Obs.get_snr(nsflux))
            return np.mean(SNs)
        else:
            print("not list")
            return Obs.get_snr(fluxs)


    def get_snr2nl_fn(self, flux_in_res, step):
        #-----------------------------------------
        # choose the noise levels so that the S/N 
        # comes at around the predetermined levels
        #-----------------------------------------
        # self.noise_level_grid = [2,5,10,20,30,40,50,100,200]

        var_in_res = Obs.get_var(flux_in_res, self.sky_in_res, step=step)
        noise      = Obs.get_noise(var_in_res)

        SN = []
        for noise_level in self.noise_level_grid:
            ssobs = flux_in_res + noise_level * noise
            sn    = Obs.get_snr(ssobs)
            SN.append(sn)
        print("snr2nl-SN", SN)
        f = sp.interpolate.interp1d(SN, self.noise_level_grid, fill_value=0)
        return f



    # def make_nlList(self, flux_H, skym, step=5):
    #     #-----------------------------------------
    #     # choose the noise levels so that the S/N 
    #     # comes at around the predetermined levels
    #     #-----------------------------------------
        
    #     if self.snrList is None:
    #         self.snrList = [11,22,33,55,110]
    #     if self.noise_level_grid is None:
    #         self.noise_level_grid = [2,5,10,20,50,100,200,500]



    #     # ssm   = Util.getModel(ss,0)
    #     ssm   = Obs.resampleFlux_i(flux_H, step)       
    #     varm  = Obs.get_var(ssm,skym)
    #     noise = Obs.get_noise(varm)  

    #     SN = []
    #     for noise_level in self.noise_level_grid:
    #         ssobs = ssm + noise_level * noise
    #         sn    = Obs.get_snr(ssobs)
    #         SN.append(sn)
    #     f = sp.interpolate.interp1d(SN, self.noise_level_grid, fill_value=0)
        
    #     noise_level_interpd = f(self.snrList)  
    #     return noise_level_interpd


    
    @staticmethod
    def get_var(ssm, skym, step=1):
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
        if step <= 1: 
            return varm
        else:
            return np.divide(varm, step)

    @staticmethod
    def get_noise_N(varm, N):
        out = np.zeros((N, varm.shape[0]))
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

    