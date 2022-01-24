import numpy as np
import pandas as pd
from .util import Util
from .baseplot import BasePlot
from PIML.obs.obs import Obs


class BaseBox(Util):


    def __init__(self):
        super().__init__()
        self.PLT = BasePlot()


# init ------------------------------------------------------------------------
    def init_W(self, W):
        self.W = W
        self.Ws = BaseBox.DWs[W]

    def get_flux_in_Wrange(self, wave, flux):
        return Obs._get_flux_in_Wrange(wave, flux, self.Ws)

    def init_WR(self, W, R):
        self.init_W(W)
        if isinstance(R, list):
            self.init_Rs(R)
        elif isinstance(R, str):
            self.init_R(R)
        else:
            raise ValueError('R must be a string or a list of strings')

    def init_R(self, R):
        self.R = R
        self.RR = BaseBox.DRR[R]
        self.PhyMin, self.PhyMax, self.PhyRng, self.PhyNum, self.PhyMid = self.get_bnd(R)
        self.minmax_scaler, self.minmax_rescaler = BaseBox.get_minmax_scaler_fns(self.PhyMin, self.PhyRng)
        self.pmt2pdx_scaler, _ = BaseBox.get_pdx_scaler_fns(self.PhyMin)

    def init_Rs(self, Rs):
        self.Rs = Rs
        self.RR = [BaseBox.DRR[R] for R in Rs]
        self.nR = len(Rs)
        self.init_bnds(Rs)
        self.init_scalers()

    def init_bnds(self, Rs=None):
        self.DPhyMin = {}
        self.DPhyMax = {}
        self.DPhyRng = {}
        self.DPhyNum = {}
        self.DPhyMid = {}

        if Rs is None: Rs = BaseBox.Rnms
        for R in Rs:
            PhyMin, PhyMax, PhyRng, PhyNum, PhyMid = self.get_bnd(R)
            self.DPhyMin[R] = PhyMin
            self.DPhyMax[R] = PhyMax
            self.DPhyRng[R] = PhyRng
            self.DPhyNum[R] = PhyNum
            self.DPhyMid[R] = PhyMid

    def init_scalers(self):
        self.minmax_scaler = {}
        self.minmax_rescaler = {}
        self.pmt2pdx_scaler = {}

        for R, PhyMin in self.DPhyMin.items():
            self.minmax_scaler[R], self.minmax_rescaler[R] = BaseBox.get_minmax_scaler_fns(PhyMin, self.DPhyRng[R])
            self.pmt2pdx_scaler[R], _ = BaseBox.get_pdx_scaler_fns(PhyMin)

    
    def init_plot_R(self):
        self.PLT.make_box_fn = lambda x: self.PLT.box_fn(self.PhyRng, self.PhyMin, 
                                                        self.PhyMax, n_box=x, 
                                                        c=BaseBox.DRC[self.R], RR=self.RR)

# dataloader ------------------------------------------------------------------
    def load_data(self, Res, RR, step):
        wave_H, flux_H, self.pdx, self.para = self.IO.load_bosz(Res, RR=self.RR)
        self.pdx0 = self.pdx - self.pdx[0]


    def prepare_data(self, Res, R, step):
        wave_H, flux_H, pdx, para = self.IO.load_bosz(Res, RR=BaseBox.DRR[R])
        pdx0 = pdx - pdx[0]
        wave_H, flux_H = self.get_flux_in_Wrange(wave_H, flux_H)
        self.wave, flux = Obs.resample(wave_H, flux_H, step)
        self.init_obs(wave_H, step)









#Obs --------------------------------------------------------------------------
    def init_obs(self, wave_H, step):
        self.Obs = Obs()
        self.init_sky(wave_H, step)

    def init_sky(self, wave_H, step, flux0=None):
        self.Obs.prepare_sky(wave_H, step)
        if flux0 is not None:
            self.Obs.prepare_snr(flux0)





# static ----------------------------------------------------------------------



    @staticmethod
    def init_para(para):
        return pd.DataFrame(para, columns=BaseBox.PhyShort)

    @staticmethod
    def get_bnd(R):
        bnd = np.array(BaseBox.DRs[R])
        PhyMin, PhyMax = bnd.T
        PhyRng = np.diff(bnd).T[0]
        PhyNum = PhyRng / BaseBox.PhyTick 
        PhyMid = (PhyNum //2) * BaseBox.PhyTick + PhyMin
        return PhyMin, PhyMax, PhyRng, PhyNum, PhyMid

    @staticmethod
    def get_minmax_scaler_fns(PhyMin, PhyRng):
        def scaler_fn(x):
            return (x - PhyMin) / PhyRng
        def inverse_scaler_fn(x):
            return x * PhyRng + PhyMin        
        return scaler_fn, inverse_scaler_fn

    @staticmethod
    def get_pdx_scaler_fns(PhyMin):
        def scaler_fn(x):
            return np.divide((x - PhyMin) ,BaseBox.PhyTick)
        def inverse_scaler_fn(x):
            return x * BaseBox.PhyTick + PhyMin
        return scaler_fn, inverse_scaler_fn

    @staticmethod
    def get_bdx_R(R, dfpara=None, bnds=None, cutCA = False):
        #TODO get range index
        if dfpara is None: dfpara = Util.IO.read_dfpara()
        if bnds is None: 
            bnds = BaseBox.DRs[R]
        Fs, Ts, Gs, Cs, As  = bnds

        maskM = (dfpara["M"] >= Fs[0]) & (dfpara["M"] <= Fs[1]) 
        maskT = (dfpara["T"] >= Ts[0]) & (dfpara["T"] <= Ts[1]) 
        maskL = (dfpara["G"] >= Gs[0]) & (dfpara["G"] <= Gs[1]) 
        mask = maskM & maskT & maskL
        if cutCA:
            maskC = (dfpara["C"] >= Cs[0]) & (dfpara["C"] <= Cs[1])
            maskA = (dfpara["A"] >= As[0]) & (dfpara["A"] <= As[1])
            mask = mask & maskC & maskA

        return dfpara[mask].index

    # def load_dfpara(self):
    #     dfpara = IO.read_dfpara()
    #     return dfpara

    @staticmethod
    def get_bdx(dfpara=None, para=None):
        if dfpara is None: 
            if para is None:
                dfpara = Util.IO.read_dfpara()
            else:
                dfpara = BaseBox.init_para(para)

        DBdx = {}
        for R in BaseBox.Rnms:
            bdx = BaseBox.get_bdx_R(R, dfpara, bnds=BaseBox.DRs[R], cutCA=False)
            DBdx[R] = bdx
        return DBdx


    @staticmethod
    def get_flux_para_R(R, flux, para, DBdx=None):
        if DBdx is None: DBdx = BaseBox.get_bdx(para=para)
        bdx = DBdx[R]
        boxFlux = flux[bdx]
        boxPara = para[bdx]
        print(boxFlux.shape, boxPara.shape)
        return bdx, boxFlux, boxPara

    @staticmethod
    def box_data(wave, flux, pdx, para, DBdx=None, Res=None, Rs=None, out=False, save=True):
        if save and (Res is None):
            raise ValueError("Res is None")
        if out:
            boxFluxs, boxPdxs, boxParas = {}, {} ,{}
        if DBdx is None: DBdx = BaseBox.get_bdx(para=para)
        if Rs is None: Rs = BaseBox.Rnms
        for R in Rs:
            boxFlux, boxPdx, boxPara = BaseBox.box_data_R(R, wave, flux, pdx, para, DBdx, Res, save)
            if out: 
                boxFluxs[R] = boxFlux
                boxPdxs[R] = boxPdx
                boxParas[R] = boxPara
        if out:
            return boxFluxs, boxPdxs, boxParas

    @staticmethod
    def box_data_R(R, wave, flux, pdx, para, DBdx=None, Res=None, save=False):
        bdx, boxFlux, boxPara = BaseBox.get_flux_para_R(R, flux, para, DBdx=DBdx)
        boxPdx = pdx[bdx] if pdx is not None else None
        RR = BaseBox.DRR[R]
        if save: Util.IO.save_bosz_box(Res, RR, wave, boxFlux, boxPdx, boxPara, overwrite=1)
        return boxFlux, boxPdx, boxPara

    @staticmethod
    def get_random_pmt(PhyRng, PhyMin, N_pmt):
        pmt0 = np.random.uniform(0,1,(N_pmt,5))
        pmts = pmt0 * PhyRng + PhyMin   
        return pmts