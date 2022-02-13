import os
import numpy as np
from tensorflow.keras.models import load_model

from PIML.nn.dnn.plotdnn import PlotDNN
from PIML.nn.trainboxW import TrainBoxW


class EvalBoxW(TrainBoxW):
    def __init__(self) -> None:
        super().__init__()
        self.pRngs = {}
        self.pMins = {}
        self.pMaxs = {}
        self.f_test = None
        self.testNL = None
        self.DPpred = {}
        self.DXtest = None

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
        self.setup_scalers(odx)

        self.nOdx = len(odx)
        #FIXME 
        self.load_eigv(self.topk, stack=1)
        if isinstance(model_names, str): model_names = [model_names]
        for model_name in model_names:
            self.set_model_R0(model_name)

    def set_scale_predict_R0(self, R0):
        self.nn[R0].nn_rescaler = lambda x: self.rescale(x, R0)

    def test(self, testNL=None, nTest=100, pmts=None, new=True, seed=922):
        self.setup_scalers(self.odx)        
        self.nTest = nTest
        self.testNL = testNL or self.trainNL
        self.nnRs = list(self.nn.keys())

        if self.model_names is None:
            [self.set_scale_predict_R0(R0) for R0 in self.nnRs]

        if self.f_test is None or new:
            self.f_test, self.p_test = self.prepare_testset(self.nTest, pmts=pmts, noise_level=self.testNL, eigv=None, odx=self.odx)

        self.DXtest = self.project_flux_on_PC() if self.onPCA else self.f_test

        for R0 in self.nnRs:
            self.DPpred[R0] = self.test_R0(R0)
        

        vertical = 1 if self.nR > 2 else 0
        self.init_eval()
        self.eval(self.DPpred, self.p_test, vertical=vertical)
        
    def project_flux_on_PC(self):
        DXtest = {}
        if isinstance(self.eigv, np.ndarray):
            for R0, model in self.nn.items():
                assert np.allclose(model.eigv, self.eigv)
                DXtest_R0 = self.f_test[R0].dot(self.eigv.T)      
                DXtest[R0] = DXtest_R0

        elif isinstance(self.eigv, dict):
            DXtest = {}
            for R0, model in self.nn.items():
                DXtest_R0 = {}
                eigv = self.nn[R0].eigv
                for R1, flux in self.f_test.items():
                    DXtest_R0[R1] = flux.dot(eigv)

                DXtest[R0] = DXtest_R0
        return DXtest

    def test_R0(self, R0):
        DPpred_R0 = {}
        for R1, flux in self.DXtest.items():
            if isinstance(flux, np.ndarray):
                DPpred_R0[R1] = self.nn[R0].scale_predict(flux)
            elif isinstance(flux, dict):
                pass
                #FIXME
        return DPpred_R0

    #eval ---------------------------------------------------------------------------------
    def init_eval(self):
        self.PhyLong =  [EvalBoxW.PhyLong[odx_i] for odx_i in self.odx]
        self.init_plot()
        # snr = self.estimate_snr(self.testNL)
        # pmts = self.get_random_pmt(10)
        # self.eval_pmts_noise(pmts, self.testNL, nObs=100, n_box=0.2)


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
        #     self.test(testNL=self.trainNL, nTest=100)

    def eval(self, DPpred, p_test, vertical=0):
        print(self.model_names)
        self.eval_acc(DPpred, p_test, vertical=vertical)
        if len(DPpred) > 1: 
            self.eval_cross(DPpred)

    def eval_acc(self, DPpred, p_test, vertical=False):
        for R0 in DPpred.keys():
            self.eval_acc_R0(R0, DPpred, p_test, vertical=vertical)

    def eval_acc_R0(self, R0, DPpred, p_test, snr=None, vertical=False):
        f, axs = self.PLT.plot_acc(DPpred[R0][R0], p_test[R0], self.pMins[R0], self.pMaxs[R0], 
                                    RR=EvalBoxW.DRR[R0], c1=EvalBoxW.DRC[R0], axes_name = self.PhyLong, vertical=vertical)
        if snr is None:
            f.suptitle(f"NL = {self.testNL}")
        else:
            f.suptitle(f"SNR = {snr:.2f}")

    def eval_cross(self, DPpred, n_box=1, snr=None):
        crossMat = self.PLT.get_crossMat(DPpred)
        for R0 in self.nnRs:
            self.eval_cross_R0(R0, crossMat=crossMat, n_box=n_box, snr=snr)

    def eval_cross_R0(self, R0, crossMat=None, n_box=0.2, snr=None):
        fns = []
        if crossMat is not None:
            R0_idx = self.nnRs.index(R0)
            crossMat_R0 = crossMat[R0_idx]
        for ii, R1 in enumerate(self.Rs):
            lgd = None if crossMat is None else crossMat_R0[ii]   
            fns = fns + [self.PLT.scatter_fn(self.DPpred[R0][R1], c=EvalBoxW.DRC[R1], s=1, lgd=lgd)]

        f = self.PLT.plot_box_R0(R0, fns = fns, n_box=n_box)
        if snr is None:
            f.suptitle(f"NL = {self.testNL}")
        else:
            f.suptitle(f"SNR = {snr:.2f}")

    def eval_traj(self, R0, DPpred, n_box=0.1):
        fns = []
        # BoxW.DRC[R0]
        fns = fns + [self.PLT.line_fn(DPpred, c="r", lgd=None)]

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

#