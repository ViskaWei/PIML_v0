import os
from tensorflow.keras.models import load_model

from PIML.nn.dnn.plotdnn import PlotDNN
from PIML.nn.trainboxW import TrainBoxW


class EvalBoxW(TrainBoxW):
    def __init__(self) -> None:
        super().__init__()
        self.pRngs = {}
        self.pMins = {}
        self.pMaxs = {}
        self.x_test = None
        self.test_NL = None
        self.p_pred = {}
        self.model_names = None

    def load_model(self, model_name, path=None):
        if path is None:
            path = os.path.join(TrainBoxW.MODEL_DIR, model_name, "model.h5")
        model = load_model(path)
        return model

    def set_model_R0(self, model_name):
        R0 = model_name[:1]
        model = self.load_model(model_name)
        def _predict(x):
            y = model.predict(x)
            return self.rescale(y, R0)            
        model.scale_predict = _predict
        self.nn[R0] = model


    def load_train(self, model_names, odx=[0,1,2], topk=10):
        self.model_names = model_names
        self.topk = topk
        self.odx  = odx
        self.nOdx = len(odx)
        self.load_eigv(self.topk, stack=1)
        if isinstance(model_names, str): model_names = [model_names]
        for model_name in model_names:
            self.set_model_R0(model_name)

    def test_R0(self, R0, x_test):
        model = self.nn[R0]
        p_pred = {}
        for R, x_test in self.x_test.items():
            p_pred[R] = model.scale_predict(x_test)
        return p_pred

    def set_scale_predict_R0(self, R0):
        self.nn[R0].nn_rescaler = lambda x: self.rescale(x, R0)

    def test(self, test_NL=None, nTest=100, pmts=None, new=True, seed=922):
        self.nTest = nTest
        self.test_NL = test_NL
        self.nnRs = list(self.nn.keys())

        if self.model_names is None:
            [self.set_scale_predict_R0(R0) for R0 in self.nnRs]

        if self.x_test is None or new:
            self.x_test, self.p_test = self.prepare_testset(self.nTest, pmts=pmts, noise_level=test_NL, seed=seed, odx=self.odx)
        
        for R0 in self.nnRs:
            self.p_pred[R0] = self.test_R0(R0, self.x_test) 
        vertical = 1 if self.nR > 2 else 0
        self.eval(self.p_pred, self.p_test, vertical=vertical)

    def init_test(self, model_names=None, topk=10, odx=[0,1,2], mtype="DNN", save=1, train_NL=None, nTrain=1000, name=""):
        self.setup_scalers(odx)
        if model_names is None:
            self.model_names = None
            self.init_train(odx=odx, mtype=mtype, save=save, train_NL=train_NL, nTrain=nTrain, name=name)
        else:
            self.load_train(model_names, odx=odx, topk=topk)
        self.init_eval()

        


    #eval ---------------------------------------------------------------------------------
    def init_eval(self):
        self.PhyLong =  [EvalBoxW.PhyLong[odx_i] for odx_i in self.odx]
        self.init_plot()
        # snr = self.estimate_snr(self.test_NL)
        # pmts = self.get_random_pmt(10)
        # self.eval_pmts_noise(pmts, self.test_NL, nObs=100, n_box=0.2)


    def init_plot(self):
        self.PLT = PlotDNN(self.odx)
        self.PLT.Rs = self.Rs
        self.PLT.RRs = self.RRs
        self.PLT.pMaxs = self.pMaxs
        self.PLT.pMins = self.pMins
        self.PLT.pRngs = self.pRngs
        self.PLT.PhyLong = self.PhyLong
        # self.PLT.make_box_fn = lambda x: self.PLT.box_fn(self.pRng, self.pMin, 
        #                                                 self.pMax, n_box=x, 
        #                                                 c=BaseBox.DRC[self.R], RR=self.RR)

        # if eval:
        #     self.init_eval()
        #     self.test(test_NL=self.train_NL, nTest=100)

    def eval(self, p_pred, p_test, vertical=0):
        print(self.model_names)
        self.eval_acc(p_pred, p_test, vertical=vertical)
        if len(p_pred) > 1: 
            self.eval_cross(p_pred)

    def eval_acc(self, p_pred, p_test, vertical=False):
        for R0 in p_pred.keys():
            self.eval_acc_R0(R0, p_pred, p_test, vertical=vertical)

    def eval_acc_R0(self, R0, p_pred, p_test, snr=None, vertical=False):
        f, axs = self.PLT.plot_acc(p_pred[R0][R0], p_test[R0], self.pMins[R0], self.pMaxs[R0], 
                                    RR=EvalBoxW.DRR[R0], c1=EvalBoxW.DRC[R0], axes_name = self.PhyLong, vertical=vertical)
        if snr is None:
            f.suptitle(f"NL = {self.test_NL}")
        else:
            f.suptitle(f"SNR = {snr:.2f}")

    def eval_cross(self, p_pred, n_box=1, snr=None):
        crossMat = self.PLT.get_crossMat(p_pred)
        for R0 in self.nnRs:
            self.eval_cross_R0(R0, crossMat=crossMat, n_box=n_box, snr=snr)

    def eval_cross_R0(self, R0, crossMat=None, n_box=0.2, snr=None):
        fns = []
        if crossMat is not None:
            R0_idx = self.nnRs.index(R0)
            crossMat_R0 = crossMat[R0_idx]
        for ii, R1 in enumerate(self.Rs):
            lgd = None if crossMat is None else crossMat_R0[ii]   
            fns = fns + [self.PLT.scatter_fn(self.p_pred[R0][R1], c=EvalBoxW.DRC[R1], s=1, lgd=lgd)]

        f = self.PLT.plot_box_R0(R0, fns = fns, n_box=n_box)
        if snr is None:
            f.suptitle(f"NL = {self.test_NL}")
        else:
            f.suptitle(f"SNR = {snr:.2f}")

    def eval_traj(self, R0, p_pred, n_box=0.1):
        fns = []
        # BoxW.DRC[R0]
        fns = fns + [self.PLT.line_fn(p_pred, c="r", lgd=None)]

        f = self.PLT.plot_box_R0(R0, fns = fns, n_box=n_box)


# scaler ---------------------------------------------------------------------------------
    def setup_scalers(self, odx=None):
        if odx is None: odx = self.odx
        for R in self.Rs:
            self.pRngs[R], self.pMins[R], self.pMaxs[R] = self.get_scaler(R, odx)


    def get_scaler(self, R, odx):
        pRng = self.DPhyRng[R][odx]
        pMin = self.DPhyMin[R][odx]
        pMax = self.DPhyMax[R][odx]
        return pRng, pMin, pMax

    def scale(self, pval, R):
        pnorm = (pval - self.pMins[R]) / self.pRngs[R]        
        return pnorm

    def rescale(self, pnorm, R):
        pval = pnorm * self.pRngs[R] + self.pMins[R]
        return pval

