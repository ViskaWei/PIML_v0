import numpy as np
import scipy as sp

from .util import Util
class BaseSpec(Util):
    """
    Base class for all specifications.
    """
    def __init__(self):
        super().__init__()
    

    @staticmethod
    def init_W(W):
        Ws = Util.DWs[W]
        return Ws

    @staticmethod
    def _get_flux_in_Wrange(wave, flux, Ws):
        start = np.digitize(Ws[0], wave)
        end = np.digitize(Ws[1], wave)
        return wave[start:end], flux[:, start:end]

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

# log norm --------------------------------------------------------------
    @staticmethod
    def safe_log(x):
        return np.log(np.where(x <= 1, 1, x))

    @staticmethod
    def normlog_flux(fluxs):
        logflux = np.log(np.where(fluxs <= 1, 1, fluxs))
        normlogflux = logflux - logflux.mean(1)[:,None]
        return normlogflux

    @staticmethod
    def norm_flux(fluxs, log=1):
        if len(fluxs.shape) == 1: 
            fluxs = fluxs[:,None]
        if log:
            normflux = fluxs - fluxs.mean(1)[:,None]
        else:
            normflux = fluxs / fluxs.mean(1)[:,None]
        if len(fluxs.shape) == 1: 
            normflux = normflux[0]
        return normflux

    @staticmethod
    def normlog_flux_i(flux):
        logflux = np.log(np.where(flux <= 1, 1, flux))
        normlogflux = logflux - logflux.mean()
        return normlogflux

    @staticmethod
    def lognorm_flux(fluxs):
        fluxs = np.where(fluxs>0, fluxs, 1e-20)
        norm_flux = np.divide(fluxs, fluxs.mean(1)[:,None])
        # norm_flux = np.where(norm_flux <= 0, 0, norm_flux)
        lognormflux = np.log(norm_flux)
        return lognormflux

    # @staticmethod
    # def safe_log(x):
    #         a = np.exp(args[0]) if args is not None else 1e-10
    #         return np.log(np.where(x < a, a, x))

    @staticmethod
    def lognorm_flux_i(flux):
        return np.log(np.divide(flux, flux.mean()))


# resample ------------------------------------------------------------------------------

    @staticmethod
    def resampleWave(wave, step=5, verbose=1):
        if step==0:
            wave1 = wave
        else:
            #-----------------------------------------------------
            # resample the wavelengths by a factor step
            #-----------------------------------------------------
            w = np.cumsum(np.log(wave))
            b = list(range(1, len(wave), step))
            db = np.diff(w[b])
            dd = (db/step)
            wave1 = np.exp(dd) 
        if verbose: BaseSpec.print_res(wave1)
        return wave1

    @staticmethod
    def resampleFlux_i(flux, step=5):
        if step==0: return flux
        #-----------------------------------------------------
        # resample the spectrum by a factor step
        #-----------------------------------------------------
        c = np.cumsum(flux)
        b = list(range(1, len(flux), step))
        db = np.diff(c[b])
        dd = (db/step)
        return dd

    @staticmethod
    def resampleFlux(fluxs, L,step=5):
        if step == 0: return fluxs
        out = np.zeros((len(fluxs), L))
        for ii, flux in enumerate(fluxs):
            out[ii] = BaseSpec.resampleFlux_i(flux, step=step)
        return out

    @staticmethod
    def resampleSky(f, wave_grid, step=5, avg=0):
        if step==0: 
            sky_new = np.diff(f(wave_grid))
            sky_new = np.insert(sky_new, 0, f(wave_grid[0]))
        else:
            b = list(range(1,wave_grid.shape[0],step))
            sky_new = np.diff(f(wave_grid[b]))
        if avg:
            sky_new = sky_new / step
        # assert sky_new.shape == wave_grid.shape
        return sky_new

    @staticmethod
    def resample_sky(sky_H, wave_M, step=5, avg=0):
        sky_cumsum = np.cumsum(sky_H)
        b = list(range(1, wave_M, step))
        sky_new = np.diff(sky_cumsum[b])
        if avg:
            sky_new = sky_new / step
        # assert sky_new.shape == wave_grid.shape
        return sky_new

    @staticmethod
    def interp_sky_fn(sky):
        ws = sky[:,0]
        cs = np.cumsum(sky[:,1])
        f = sp.interpolate.interp1d(ws,cs, fill_value=0)
        return f

    @staticmethod
    def resample(wave, fluxs, step=10, verbose=1):
        waveL= BaseSpec.resampleWave(wave, step=step, verbose=verbose)
        L = len(waveL)
        fluxL =BaseSpec.resampleFlux(fluxs, L, step=step)
        return waveL, fluxL

    @staticmethod
    def resample_ns(wave, fluxs, errs, step=10, verbose=1):
        waveL= BaseSpec.resampleWave(wave, step=step, verbose=verbose)
        L = len(waveL)
        fluxL =BaseSpec.resampleFlux(fluxs, L, step=step)
        errL = BaseSpec.resampleFlux(errs, L, step=step)
        return waveL, fluxL, errL

    @staticmethod
    def print_res(wave):
        dw = np.mean(np.diff(np.log(wave)))
        print(f"#{len(wave)} R={1/dw:.2f}")

# --------------------------------------------------------------------------------- 
# ---------------------------------------------------------------------------------
