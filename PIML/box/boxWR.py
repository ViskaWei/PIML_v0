
import numpy as np
from PIML.util.basebox import BaseBox
from PIML.obs.obs import Obs
from PIML.method.rbf import RBF
import matplotlib.pyplot as plt


class BoxWR(BaseBox):
    def __init__(self):
        super().__init__()
        self.W = None
        self.R = None
        self.PhyMin = None
        self.PhyMax = None
        self.PhyRng = None
        self.PhyNum = None
        self.PhyMid = None
        self.Obs = Obs()
        self.RBF = RBF()

    def init_R(self, R):
        self.R = R
        self.RR = BaseBox.DRR[R]
        self.PhyMin, self.PhyMax, self.PhyRng, self.PhyNum, self.PhyMid = self.get_bnd(R)
        self.scaler, self.rescaler = BaseBox.get_scaler_fns(self.PhyMin, self.PhyRng)
        self.rbf_scaler, _ = BaseBox.get_pdx_scaler_fns(self.PhyMin)

    def init_W(self, W):
        self.W = W
        self.Ws = Obs.init_W(self.W)

    def get_flux_in_Wrange(self, wave, flux):
        return Obs._get_flux_in_Wrange(wave, flux, self.Ws)

    def init(self, W, R, Res, step):
        self.init_W(W)
        self.init_R(R)
        self.Res = Res
        self.step = step
        self.load_data(Res, self.RR)
    
    def load_data(self, Res, RR):
        wave, flux, self.pdx, self.para = self.IO.load_bosz(Res, RR=RR)
        self.pdx0 = self.pdx - self.pdx[0]

        self.wave, self.flux = self.prepro(wave, flux)
        self.build_rbf(self.flux)

    def get_grid_model(self, pmt):
        fdx = self.Obs.get_fdx_from_pmt(pmt, self.para)
        return self.flux[fdx]

    def prepro(self, wave, flux):
        self.wave0, flux = self.get_flux_in_Wrange(wave, flux)
        self.Obs.getSky(self.wave0, self.step)
        wave, flux = Obs.resample(self.wave0, flux, self.step)
        return wave, flux

    def build_rbf(self, flux):
        logflux = self.Obs.safe_log(flux)
        self.RBF._build_rbf(self.pdx0, logflux)
        self.RBF.rbf_scaler = self.rbf_scaler
        self.RBF.pred_scaler = np.exp

    def test_rbf(self, pmt1, pmt2, pmt=None):
        flux1, flux2 = self.get_grid_model(pmt1), self.get_grid_model(pmt2)
        if pmt is None: pmt = 0.5 * (pmt1 + pmt2)
        interpFlux = self.RBF.rbf_predict([pmt])
        plt.plot(self.wave, interpFlux, label= pmt)
        plt.plot(self.wave, flux1, label = pmt1)
        plt.plot(self.wave, flux2, label = pmt2)
        plt.legend()



