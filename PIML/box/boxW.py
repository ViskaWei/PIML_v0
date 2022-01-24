import os
import h5py
import numpy as np
from PIML import obs
from PIML.util.basebox import BaseBox
from PIML.util.baseplot import BasePlot

from PIML.obs.obs import Obs
from PIML.method.llh import LLH
from PIML.method.rbf import RBF
from PIML.method.bias import Bias
import matplotlib.pyplot as plt
from tqdm import tqdm

from PIML.util.util import Util

# class testBoxWR(BoxWR):


class BoxWR(BaseBox):
    def __init__(self):
        super().__init__()
        self.W = None
        self.R = None
        self.Obs = Obs()
        self.LLH = LLH()
        self.PLT = BasePlot()
        self.topk = None



    def init(self, W, Rs, Res, step, topk=10, onPCA=1):
        self.init_WR(W,Rs)
        self.Res = Res
        self.step = step
        self.onPCA = onPCA
        self.topk = topk
