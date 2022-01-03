import numpy as np
from scipy.interpolate import RBFInterpolator

class RBF(object):
    def __init__(self, coord, coord_scaler=None):
        self.rbf = None
        self.coord = coord
        self.coord_scaler = coord_scaler

        if coord_scaler is None:
            self.coord_scaler = lambda x: x

    def train_rbf(self, coord, val):
        print(f"Building RBF with gaussan kernel on data shape {val.shape}")
        rbf_interpolator = RBFInterpolator(coord, val, kernel='gaussian', epsilon=0.5)
        return rbf_interpolator

    def interp_scaler(self, x):
        return x

    def build_rbf(self,  rbf_interpolator,  interp_scaler=None):
        if interp_scaler is None: interp_scaler = self.interp_scaler
        def rbf(x):
            flag = False
            if x.ndim == 1: 
                x = [x]
                flag = True
            x_scale = self.coord_scaler(x)
            interp = rbf_interpolator(x_scale)
            out = interp_scaler(interp)
            if flag: 
                return out[0]
            else:
                return out
        return rbf

    def build_logflux_rbf_interp(self, logflux):
        rbf_interp = self.train_rbf(self.coord, logflux)
        rbf = self.build_rbf(rbf_interp, interp_scaler=np.exp)
        def interp_logflux(x):
            return rbf(x)
        return interp_logflux

    def build_PC_rbf_interp(self, logA, pcflux, eigv):
        rbf_interp_logA = self.train_rbf(self.coord, logA)
        rbf_logA = self.build_rbf(rbf_interp_logA)
        rbf_interp_ak = self.train_rbf(self.coord, pcflux)
        rbf_ak = self.build_rbf(rbf_interp_ak)
        
        def interp_obs_fn(pmt, log=0):
            logA = rbf_logA(pmt)
            ak = rbf_ak(pmt)
            logModel = ak.dot(eigv)
            logAModel = logModel + logA
            if log: 
                return logAModel
            else:
                return np.exp(logAModel)

        def interp_model_fn(pmt, log=0):
            ak = rbf_ak(pmt)
            logModel = ak.dot(eigv)
            if log: 
                return logModel
            else:
                return np.exp(logModel)

        return interp_obs_fn, interp_model_fn, rbf_logA, rbf_ak

