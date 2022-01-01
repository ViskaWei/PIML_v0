import numpy as np
import pandas as pd
from .constants import Constants
from .IO import IO

class BaseBox(Constants):
    """ Box Constants """
    DRs =  {"M": [[-2.5, 0.0], [3500, 5000], [0.0, 1.5],[-0.75, 0.5], [-0.25, 0.5]], 
            "W": [[-2.0, 0.0], [5500, 7500], [3.5, 5.0],[-0.75, 0.5], [-0.25, 0.5]],
            "C": [[-2.0, 0.0], [3750, 5500], [3.5, 5.0],[-0.75, 0.5], [-0.25, 0.5]], 
            "B": [[-2.5,-1.5], [6750, 9500], [2.0, 3.5],[-0.75, 0.5], [-0.25, 0.5]],
            "R": [[-1.0, 0.0], [5500, 6750], [2.0, 3.5],[-0.75, 0.5], [-0.25, 0.5]], 
            "G": [[-2.5,-1.0], [4000, 5500], [1.5, 3.5],[-0.75, 0.5], [-0.25, 0.5]]}
    DRR = {"M": "M31G"  ,"W":"MWW",       "C":"MWC",  "B":"BHB",       "R":"RHB","G":"DGG"}
    DRC = {"M": "orange","W":"lightgreen","C":"brown","B":"dodgerblue","R":"red","G":"fuchsia"}

    Rnms = list(DRR.keys())
    RRnms = list(DRR.values())

    def __init__(self):
        super().__init__()
        self.IO = IO()


    def init_bnds(self):
        self.DPhyMin = {}
        self.DPhyMax = {}
        self.DPhyRng = {}
        self.DPhyNum = {}
        self.DPhyMid = {}
        for R in BaseBox.Rnms:
            PhyMin, PhyMax, PhyRng, PhyNum, PhyMid = self.get_bnd(R)
            self.DPhyMin[R] = PhyMin
            self.DPhyMax[R] = PhyMax
            self.DPhyRng[R] = PhyRng
            self.DPhyNum[R] = PhyNum
            self.DPhyMid[R] = PhyMid

    def init_R(self, R):
        self.R = R
        self.RR = BaseBox.DRR[R]
        self.PhyMin, self.PhyMax, self.PhyRng, self.PhyNum, self.PhyMid = self.get_bnd(R)
        self.minmax_scaler, self.minmax_rescaler = BaseBox.get_minmax_scaler_fns(self.PhyMin, self.PhyRng)
        self.pmt2pdx_scaler, _ = BaseBox.get_pdx_scaler_fns(self.PhyMin)

    def init_W(self, W):
        self.W = W
        self.Ws = BaseBox.DWs[W]

    def init_WR(self, W, R):
        self.init_W(W)
        self.init_R(R)



    @staticmethod
    def init_para(para):
        return pd.DataFrame(para, columns=Constants.PhyShort)

    @staticmethod
    def get_bnd(R):
        bnd = np.array(BaseBox.DRs[R])
        PhyMin, PhyMax = bnd.T
        PhyRng = np.diff(bnd).T[0]
        PhyNum = PhyRng / Constants.PhyTick 
        PhyMid = (PhyNum //2) * Constants.PhyTick + PhyMin
        return PhyMin, PhyMax, PhyRng, PhyNum, PhyMid

    @staticmethod
    def get_minmax_scaler_fns(PhyMin, PhyRng):
        def scaler_fn(x):
            return (x - PhyMin) / PhyRng
        def inverse_scaler_fn(x):
            return x * PhyRng + PhyMin        
        return scaler_fn, inverse_scaler_fn

    
    def get_pdx_scaler_fns(PhyMin):
        def scaler_fn(x):
            return np.divide((x - PhyMin) ,Constants.PhyTick)
        def inverse_scaler_fn(x):
            return x * Constants.PhyTick + PhyMin
        return scaler_fn, inverse_scaler_fn

    @staticmethod
    def get_bdx_R(R, dfpara=None, bnds=None, cutCA = False):
        #TODO get range index
        if dfpara is None: dfpara = IO.read_dfpara()
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
                dfpara = IO.read_dfpara()
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
        if save: IO.save_bosz_box(Res, RR, wave, boxFlux, boxPdx, boxPara, overwrite=1)
        return boxFlux, boxPdx, boxPara

    @staticmethod
    def get_random_pmt(PhyRng, PhyMin, N_pmt):
        pmt0 = np.random.uniform(0,1,(N_pmt,5))
        pmts = pmt0 * PhyRng + PhyMin   
        return pmts