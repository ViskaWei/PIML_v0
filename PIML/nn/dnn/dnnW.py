import os
import sys
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIML.nn.dnn.dnnWR import dnnWR

from PIML.util.basebox import BaseBox

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
tf.config.list_physical_devices('GPU') 
import warnings
warnings.filterwarnings("ignore")
import logging 
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from .dnnWR import dnnWR

class dnnW(dnnWR):
    def __init__(self):
        super().__init__()


    def init(self):
        self.init_bnds()

# scaler ---------------------------------------------------------------------------------
    def setup_scalers(self, odx=None):
        if odx is None: odx = self.odx

        for R in dnnW.Rnms:
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