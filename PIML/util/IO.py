import os
import h5py
import numpy as np
import pandas as pd
from PIML.util.constants import Constants

class IO():

    
#load -----------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def read_dfpara():
        dfpara = pd.read_csv(Constants.PARA_PATH)
        return dfpara

    @staticmethod
    def load_bosz(Res, W="", RR=None, PATH=None):
        RR = "" if RR is None else "_"+RR
        if PATH is None: 
            PATH =  os.path.join(Constants.GRID_DIR, W, f"bosz_{Res}{RR}.h5")
        with h5py.File(PATH, 'r') as f:
            wave = f["wave"][:]
            flux = f["flux"][:]
            para = f["para"][:]
            if "pdx" in f.keys():
                pdx = f["pdx"][:]
            else:
                pdx = None
        return wave, flux, pdx, para

    @staticmethod
    def load_laszlo_bosz(Res, PATH=None, getPara=True, save=False, overwrite=False):
        """
        Loads the Bosz dataset.
        """
        if PATH is None: 
            PATH = os.path.join(Constants.LASZLO_GRID_DIR, f"bosz_{Res}", "spectra.h5")
        print(f"Loading Bosz dataset from {PATH}")
        with h5py.File(PATH, "r") as f:
            mask = f['flux_idx'][()]
            wave = f['wave'][()]
        print(wave)
        idx_s, idx_e = np.digitize([3000, 14000], wave)
        wave = wave[idx_s:idx_e]
        with h5py.File(PATH, "r") as f:
            flux = f['flux'][..., idx_s:idx_e]
            # flux = f['flux'][0,0,0,0,0, idx_s:idx_e]


        flux_valid = flux[mask]
        # flux_valid = np.array([flux])
        print(flux)
        print(wave.shape, flux_valid.shape)
        if getPara:
            pdx, para = IO.get_para_from_mask(mask)
            print(pdx.shape, para.shape)
            if save:
                IO.save_laszlo_bosz(Res, wave, flux_valid, pdx, para, overwrite=overwrite)
                IO.save_para(para, overwrite=overwrite)
            return wave, flux_valid, pdx, para
        return wave, flux_valid, mask

    @staticmethod
    def get_para_from_mask(mask):
        pdx =  np.array(np.where(mask))
        nSpec = pdx.shape[1]
        para=[]
        for ii, phyname in enumerate(Constants.PhyShort):
            pval = eval(f"Constants.U{phyname}")
            pdx_ii = pdx[ii]
            print(pdx_ii.shape, len(pval), pval)
            plist = [pval[pdx_ii[i]]  for i in range(nSpec)]        
            para.append(plist)
        para = np.array(para)
        print(pdx)
        print(para)
        return pdx.T, para.T


#save --------------------------------------------------------------------------------
    @staticmethod
    def save(wave, flux, pdx, para, SAVE_PATH, overwrite=0):
        if os.path.exists(SAVE_PATH):
            if overwrite:
                print(f"Overwriting {SAVE_PATH}")
            else:
                raise (f"{SAVE_PATH} exists")
        else:
            print(f"Saving dataset to {SAVE_PATH}")

        with h5py.File(SAVE_PATH, "w") as f:
            f.create_dataset('wave', data=wave, shape=wave.shape)
            f.create_dataset('flux', data=flux, shape=flux.shape)
            if pdx is not None:
                f.create_dataset('pdx', data=pdx, shape=pdx.shape)
            f.create_dataset('para', data=para, shape=para.shape)


    @staticmethod
    def save_laszlo_bosz(Res, wave, flux_valid, pdx, para, W="", overwrite=0):
        SAVE_PATH = os.path.join(Constants.GRID_DIR, f"bosz_{Res}{W}.h5")
        IO.save(wave, flux_valid, pdx, para, SAVE_PATH, overwrite)

    @staticmethod
    def save_para(para, SAVE_PATH=None, overwrite=0):
        if SAVE_PATH is None: SAVE_PATH = Constants.PARA_PATH
        if os.path.exists(SAVE_PATH):
            if overwrite:
                print(f"Overwriting {SAVE_PATH}")
            else:
                raise (f"{SAVE_PATH} exists")
        else:
            print(f"Saving dataset to {SAVE_PATH}")
        if para.shape[1] !=5: para = para.T
        assert para.shape[1] == 5
        dfpara = pd.DataFrame(data=para, columns=Constants.PhyShort)
        dfpara.to_csv(SAVE_PATH, index=0)

    @staticmethod
    def save_bosz_box(Res, RR, wave, flux, pdx, para, overwrite=0):
        SAVE_PATH = os.path.join(Constants.GRID_DIR, f"bosz_{Res}_{RR}.h5")
        IO.save(wave, flux, pdx, para, SAVE_PATH, overwrite)


    # def save_ak(self, pmt=None, SAVE_PATH=None):
    #     if pmt is None: pmt = self.PhyMid
    #     if SAVE_PATH is None: SAVE_PATH = os.path.join(self.Obs.DATA_PATH, "ak.h5")
    #     ak= self.rbf_ak(pmt)
    #     with h5py.File(SAVE_PATH, "a") as f:
    #         f.create_dataset(self.R, data=ak, shape=ak.shape)

    # def load_ak(self, pmt=None, LOAD_PATH=None):
    #     if LOAD_PATH is None: LOAD_PATH = os.path.join(self.Obs.DATA_PATH, "ak.h5")
    #     Dak = {}
    #     with h5py.File(LOAD_PATH, "r") as f:
    #         for key in f.keys():
    #             Dak[key] = f[key][:]
    #     return Dak

    # def plot_Dak(self, Dak):
    #     f, axs = plt.subplots(3,2, figsize=(16,12), facecolor="w")
    #     axs = axs.flat
    #     for ii, (key, val) in enumerate(Dak.items()):
    #         axs[ii].plot(Dak[key], 'ro', label=key)
    #         axs[ii].set_xlabel("$a_k$")