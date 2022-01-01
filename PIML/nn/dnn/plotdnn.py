import numpy as np
from tqdm import tqdm
from PIML.util.basebox import BaseBox
from PIML.util.baseplot import BasePlot
import matplotlib.pyplot as plt

class plotDNN(BasePlot):


    @staticmethod
    def plot_acc(pred, truth, pMin, pMax, RR="", c1='r', fsize=4, axes_name=None):
        nData, nFtr = pred.shape
        f, axs = plt.subplots(1, nFtr, figsize=(nFtr * fsize, fsize), facecolor="w")
        for ii, ax in enumerate(axs.T):
            x, y = truth[:,ii], pred[:,ii]
            RMS = np.sqrt(np.mean((x - y) ** 2))
            ax.scatter(x, y, s=1, alpha=0.5, color="k")
            
            ax.plot([pMin[ii], pMax[ii]], [pMin[ii], pMax[ii]], c=c1, lw=2)
            ax.set_xlim(pMin[ii], pMax[ii])
            ax.set_ylim(pMin[ii], pMax[ii])
            if axes_name is not None:
                ax_name = axes_name[ii]
                ax.annotate(f"{RR}-NN\n{ax_name}\n$\Delta${ax_name}={RMS:.2f}", xy=(0.6,0.2), xycoords="axes fraction",fontsize=fsize*3, c=c1)
                ax.set_xlabel(ax_name +"_truth")
                ax.set_ylabel(ax_name + "_pred")
        return f, axs


