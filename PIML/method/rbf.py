from scipy.interpolate import RBFInterpolator

class RBF(object):
    def __init__(self):
        self.rbf = None

    def train_rbf(self, loc, val):
        print(f"Building RBF with gaussan kernel on data shape {val.shape}")
        rbf_interpolator = RBFInterpolator(loc, val, kernel='gaussian', epsilon=0.5)
        return rbf_interpolator

    def coord_scaler(self, x):
        return x
    
    def rbf_rescaler(self, x):
        return x

    def interp_scaler(self, x):
        return x

    def build_rbf(self,  rbf_interpolator, coord_scaler=None, interp_scaler=None):
        if coord_scaler is None: coord_scaler = self.coord_scaler
        if interp_scaler is None: interp_scaler = self.interp_scaler
        def rbf(x):
            flag = False
            if x.ndim == 1: 
                x = [x]
                flag = True
            x_scale = coord_scaler(x)
            interp = rbf_interpolator(x_scale)
            out = interp_scaler(interp)
            if flag: 
                return out[0]
            else:
                return out
        return rbf


