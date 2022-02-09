import numpy as np
import scipy as sp
# from scipy.stats import qmc
from .IO import IO
from .constants import Constants
import logging


class Util(Constants):
    """
    Utility functions
    """
    def __init__(self):
        self.IO = IO()
        Util.setup_logging()
        # logging.info("Util initialized")
        

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
        if nPmt == 1:
            return sample[0]
        return sample

    @staticmethod
    def get_random_grid_pmt(para, N_pmt):
        idx = np.random.randint(0, len(para), N_pmt)
        pmts = para[idx]
        return pmts
        
    @staticmethod
    def get_fdx_from_pmt(pmt, para):
        mask = True
        for ii, p in enumerate(pmt):
            mask = mask & (para[:,ii] == p)
        try:
            idx = np.where(mask)[0][0]
            return idx
        except:
            raise("No such pmt")

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
        n1 = 1.482602 / np.sqrt(6.0)* np.median(s2)
        sn = s1 / n1
        return sn


    @staticmethod
    def safe_log(x):
        if np.min(x) < 1: logging.info("Warning: log(x) is negative")
        return np.log(np.where(x <= 1, 1, x))

    @staticmethod
    def setup_logging(logfile=None):
        logging.basicConfig()
        # logging.basicConfig(filename=f'.log', encoding='utf-8', level=logging.DEBUG)
        root = logging.getLogger()
        root.setLevel(Util.get_logging_level())
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    @staticmethod
    def get_logging_level():
        # return logging.DEBUG if self.debug else logging.INFO
        return logging.INFO
