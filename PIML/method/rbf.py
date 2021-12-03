from scipy.interpolate import RBFInterpolator

class RBF(object):
    def __init__(self):
        self.rbf = None

    def _build_rbf(self, loc, val):
        print(f"Building RBF with gaussan kernel on data shape {val.shape}")
        self.rbf = RBFInterpolator(loc, val, kernel='gaussian', epsilon=0.5)

    def rbf_scaler(self, x):
        return x
    
    def rbf_rescaler(self, x):
        return x

    def pred_scaler(self, x):
        return x

    def rbf_predict(self, x):
        x_scale = self.rbf_scaler(x)
        pred = self.rbf(x_scale)
        return self.pred_scaler(pred)

