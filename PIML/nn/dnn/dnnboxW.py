import os
import sys
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
tf.config.list_physical_devices('GPU') 
import warnings
warnings.filterwarnings("ignore")
import logging 
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from ...util.baseNN import BaseDNN


import os
import sys
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIML.box.boxW import BoxW
from PIML.util.baseNN import BaseNN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
tf.config.list_physical_devices('GPU') 
import warnings
warnings.filterwarnings("ignore")
import logging 
logging.getLogger('tensorflow').setLevel(logging.FATAL)


class DnnBoxW(BoxW):
    def __init__(self):
        self.pRngs = {}
        self.pMins = {}
        self.pMaxs = {}


    def init_dnn(self, out_idx=[0,1,2], mtype="DNN", train_NL=None, nTrain=1000, nTest=100):
        self.odx  = out_idx
        self.nOdx = len(out_idx)
        self.PhyLong =  [BoxW.PhyLong[odx_i] for odx_i in self.odx]
        self.setup_scalers(out_idx)
        self.mtype = mtype
        self.train_NL = train_NL
        self.nTrain = nTrain
        self.nTest = nTest



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

# DNN model ---------------------------------------------------------------------------------

    def prepare_model(self, mtype="DNN", train_NL=None):
        NN = BaseNN()
        self.train_NL = train_NL
        NN.set_model(mtype, noise_level=train_NL, eigv=self.eigv)
        NN.set_model_shape(self.nFtr, self.nOdx)
        return NN.model

    def set_NN_R0(self, R0):
        model = self.prepare_model(mtype=self.mtype, train_NL=self.train_NL)
        

    def train_R0(self, R0):



    def init_from_BoxW(self, BoxW, ):
        self.init_WR(BoxW.W, BoxW.R)


        def prepare_trainset(nTrain, pmts=None, noise_level=None, add_noise=True):
            x, outputs = BoxW.prepare_trainset(nTrain, pmts=pmts, noise_level=noise_level, add_noise=add_noise)
            p = outputs[:, self.odx]
            y = self.nn_scaler(p)
            return x, y, p

        def prepare_testset(nTest, pmts=None, noise_level=1, seed=None):
            x, outputs = BoxW.prepare_testset(nTest, pmts=pmts, noise_level=noise_level, seed=seed)
            p = outputs[:, self.odx]
            return x, p
        
        def prepare_noiseset(pmt, noise_level=1, nObs=1):
            x = BoxW.prepare_noiseset(pmt, noise_level, nObs)
            return x

        return prepare_trainset, prepare_testset, prepare_noiseset



