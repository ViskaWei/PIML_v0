import numpy as np
import scipy as sp
from PIML.util.basespec import BaseSpec
import matplotlib.pyplot as plt


class Obs(BaseSpec):
    def __init__(self):
        self.DATADIR = '/home/swei20/LV/data/fisher/'
        self.sky = None
        self.initSky()



    def initSky(self):
        sky = np.genfromtxt(self.DATADIR +'skybg_50_10.csv', delimiter=',')
        sky[:, 0] = 10 * sky[:, 0]
        self.sky0 = sky

    def getSky(self, wave, step):
        self.sky_in_res = BaseSpec.resampleSky(self.sky0, wave, step)

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

    # likelihood---------------------------------------------------------------------------------
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

    def estimate(self, fn, x0=None, bnds=None):
        if x0 is None: x0 = self.guessEstimation(fn)
        # print(f"x0 = {x0}")
        # print(f"bnds = {bnds}")
        out = sp.optimize.minimize(fn, x0, bounds = bnds, method="Nelder-Mead")
        if (out.success==True):
            X = out.x[0]
        else:
            X = np.nan
        return X

    def guessEstimation(self, fn):
        pass

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
    def makeNLArray(ss, skym, step=5):
        #-----------------------------------------
        # choose the noise levels so that the S/N 
        # comes at around the predetermined levels
        #-----------------------------------------
        noise_level_grid = [2,5,10,20,50,100,200,500]
        snrList = [11,22,33,55,110]
        
        # ssm   = Util.getModel(ss,0)
        ssm   = Obs.resampleFlux_i(ss, step)       
        varm  = Obs.get_var(ssm,skym)
        noise = Obs.get_noise(varm)  

        SN = []
        for noise_level in noise_level_grid:
            ssobs = ssm + noise_level * noise
            sn    = Obs.getSN(ssobs)
            SN.append(sn)
        f = sp.interpolate.interp1d(SN, noise_level_grid, fill_value=0)
        
        noise_level_interpd = f(snrList)  
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

    @staticmethod
    def getObs(sconv,skym,rv, noise_level, step=5):
        #----------------------------------------------------
        # get a noisy spectrum for a simulated observation
        #----------------------------------------------------
        # inputs
        #   sconv: the rest-frame spectrum in h-pixels, convolved
        #   skym: the sky in m-pixels
        #   rv  : the radial velocity in km/s
        #   noise_level  : the noise amplitude
        # outputs
        #   ssm : the shifted, resampled sepectrum in m-pix
        #   varm: the variance in m-pixels
        #-----------------------------------------------
        # get shifted spec and the variance
        #-------------------------------------
        ssm   = Obs.getModel(sconv, rv, step=step)
        varm  = Obs.get_var(ssm,skym)
        noise = Obs.get_noise(varm)  
        #---------------------------------------
        # add the scaled noise to the spectrum
        #---------------------------------------
        ssm = ssm + noise_level * noise
        return ssm
    


    


#plot ---------------------------------------------------------------------------------

    def plotSpec(self, wave, flux_in_res, obsflux_in_res, pmt0):
        plt.figure(figsize=(9,3), facecolor='w')
        SN = Obs.get_snr(obsflux_in_res)
        plt.plot(wave, obsflux_in_res, lw=0.2, label=f"SNR={SN:.1f}")
        plt.plot(wave, flux_in_res)
        name = BaseSpec.get_pmt_name(*pmt0)
        plt.title(f"{name}")
        plt.legend()
        plt.xlabel("Wavelength [A]")
        plt.ylabel("Flux [erg/s/cm2/A]")

    