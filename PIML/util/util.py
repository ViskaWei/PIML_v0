import numpy as np
import scipy as sp
# from scipy.stats import qmc
from .IO import IO
from .constants import Constants



class Util(Constants):
    """
    Utility functions
    """
    def __init__(self):
        self.IO = IO()
        

    @staticmethod
    def get_file_path(file_name):
        """
        Get the path of the file
        :param file_name:
        :return:
        """
        import os

    @staticmethod
    def get_pmt_name(m,t,g,c,a):
        #----------------------------------
        # get short name for the spectrum
        #----------------------------------
        fname = 'T'+ Util.fmn(t)+'G'+Util.fmn(10*g)+'M'+Util.fmt(m)+'A'+Util.fmt(a)+'C'+Util.fmt(c)
        return fname

    @staticmethod
    def fmn(x):    
        return '{:02d}'.format(np.floor(x).astype(np.int32))

    @staticmethod
    def fmt(x):
        y = np.round(np.abs(10*x)+0.2).astype(np.int32)
        z = '{:+03.0f}'.format(y).replace('+','p')
        if (np.sign(x)<0):
            z = z.replace('p','m')
        return z
    
#random sample-------------------------------------------------------
    @staticmethod
    def get_random_uniform(nPmt, nPara, scaler=None, method="halton"):
        if method == "halton":            
            # Using Halton sequence to generate more evenly spaced samples
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Halton.html

            sampler = sp.stats.qmc.Halton(d=nPara, scramble=False)
            sample = sampler.random(n=nPmt)
        else:
            sample = np.random.uniform(0, 1, size=(nPmt, nPara))
        if scaler is not None:
            sample = scaler(sample)
        return sample

    @staticmethod
    def get_random_grid_pmt(para, N_pmt):
        idx = np.random.randint(0, len(para), N_pmt)
        pmts = para[idx]
        return pmts


#sampling----------------------------------------------------------------
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

    # def get_random_pmt(self, N_pmt):
    #     pmt0 = np.random.uniform(0,1,(N_pmt,5))
    #     pmts = pmt0 * self.bnds[:,0] + self.bnds[:,2]   # [0,1] -> [0,1]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                734QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ
    #     return pmts

    @staticmethod
    def safe_log(x):
        return np.log(np.where(x <= 1, 1, x))