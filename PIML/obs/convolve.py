import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class Convolve:
    def __init__(self, wave, wref=5000, res_in=5000, res_out=3000):
        self.Ws = {"Blue": [3800, 6500, 2300], "RedL": [6300, 9700, 3000], "RedM": [7100, 8850, 5000],
                    "NIR": [9400, 12600, 4300],"BL": [3800, 6500, 1000], "RML": [7100, 8850, 1000],"NL": [9400, 12600, 1000]}
        self.wave = wave
        self.wref = wref
        self.res_in = res_in
        self.res_out = res_out
        self.kernel = None
        self.sigma_kernel = None
        self.init()

    def init(self):
        self.get_sigmas()
        self.get_kernel()

    def get_sigmas(self):
        sigma_input = self.wref / self.res_in
        sigma_output = self.wref / self.res_out
        self.sigma_kernel = np.sqrt(sigma_output**2 - sigma_input**2)

    def get_kernel(self):
        kernel_mask = (self.wref - 5 < self.wave) & (self.wave < self.wref + 5)
        kernel_wave = self.wave[kernel_mask][:-1]
        kernel_wave -= kernel_wave[kernel_wave.size // 2]
        # print(kernel_wave.shape, kernel_wave)
        kernel = self.gauss_kernel(kernel_wave, self.sigma_kernel)
        self.kernel = kernel / kernel.sum()

    def gauss_kernel(self, dwave, sigma):
        return 1.0 / np.sqrt(2 * np.pi) / sigma * np.exp(-dwave**2 / (2 * sigma**2))

    def convolve(self, flux):
        return np.convolve(self.kernel, flux, mode='same')

    def convolve_all(self, flux):
        flux_conv = np.zeros_like(flux)
        for i in tqdm(range(flux.shape[0])):
            flux_conv[i] = self.convolve(flux[i])
        return flux_conv

    def plot_convolved(self, flux, flux_conv, lb=8600, ub=8700, log=1):
        mask = (self.wave > lb) & (self.wave < ub)
        plt.figure(figsize=(8, 3), facecolor="w")
        ww = self.wave[mask]
        plt.plot(ww, flux[mask], label=f"R {self.res_in}", c="k")
        plt.plot(ww, flux_conv[mask], label=f"R {self.res_out}", c="r")
        if log: plt.yscale("log")
        plt.xlim(ww[0], ww[-1])
        plt.grid(1)
        plt.legend()
        # plt.plot(self.wave[mask][::2], flux[mask][::2])


    @staticmethod
    def correct_wave_grid(wlim, resolution):
    # BOSZ spectra are written to the disk with 3 decimals which aren't
    # enough to represent wavelength at high resolutions. This code is
    # from the original Kurucz SYNTHE to recalculate the wavelength grid.

        RESOLU = resolution
        WLBEG = wlim[0]  # nm
        WLEND = wlim[1]  # nm
        RATIO = 1. + 1. / RESOLU
        RATIOLG = np.log10(RATIO)
        
        IXWLBEG = int(np.round(np.log10(WLBEG) / RATIOLG))
        WBEGIN = 10 ** (IXWLBEG * RATIOLG)
        if WBEGIN < WLBEG:
            IXWLBEG = IXWLBEG + 1
            WBEGIN = 10 ** (IXWLBEG * RATIOLG)
            
        IXWLEND = int(np.round(np.log10(WLEND) / RATIOLG))
        WLLAST = 10 ** (IXWLEND * RATIOLG)
        if WLLAST >= WLEND:
            IXWLEND = IXWLEND - 1
            WLLAST = 10 ** (IXWLEND * RATIOLG)
        LENGTH = IXWLEND - IXWLBEG + 1
        DWLBEG = WBEGIN * RATIO - WBEGIN
        DWLLAST = WLLAST - WLLAST / RATIO
        
        a = np.linspace(np.log10(WBEGIN), np.log10(WLLAST), LENGTH)
        cwave = 10 ** a
        
        return cwave