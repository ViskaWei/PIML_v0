import numpy as np
from tqdm import tqdm
from PIML.util.basebox import BaseBox
from PIML.util.baseplot import BasePlot
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Patch
from PIML.util.util import Util



class PlotDNN(BasePlot):
    def __init__(self, odx):
        self.odx = odx
        self.nOdx = len(odx)

        self.pRngs = None
        self.pMins = None
        self.pMaxs = None
        self.PhyLong = None
        
    @staticmethod
    def plot_acc(pred, truth, pMin, pMax, RR="", c1='r', fsize=4, axes_name=None, vertical=False):
        nData, nFtr = pred.shape
        height = fsize // 2 if vertical else fsize
        f, axs = plt.subplots(1, nFtr, figsize=(nFtr * fsize, height), facecolor="w")
        for ii, ax in enumerate(axs.T):
            x, y = truth[:,ii], pred[:,ii]
            RMS = np.sqrt(np.mean((x - y) ** 2))
            if vertical: 
                y = y - x
                ax.plot([pMin[ii], pMax[ii]], [0,0], c=c1, lw=2)
            else:
                ax.plot([pMin[ii], pMax[ii]], [pMin[ii], pMax[ii]], c=c1, lw=2)

            ax.scatter(x, y, s=1, alpha=0.5, color="k")
            
            ax.set_xlim(pMin[ii], pMax[ii])
            if vertical:
                ymax = np.max(np.abs(y))
                ax.set_ylim(-ymax, ymax)
            else:
                ax.set_ylim(pMin[ii], pMax[ii])
            if axes_name is not None:
                ax_name = axes_name[ii]
                ax.set_xlabel(ax_name +"_truth")
                if vertical:
                    ax.annotate(f"{RR} - $\Delta${ax_name}={RMS:.2f}", xy=(0.1,0.85), xycoords="axes fraction",fontsize=fsize*3, c=c1)
                    ax.set_ylabel(ax_name + "_error")
                else:
                    ax.annotate(f"{RR}-NN\n{ax_name}\n$\Delta${ax_name}={RMS:.2f}", xy=(0.6,0.2), xycoords="axes fraction",fontsize=fsize*3, c=c1)
                    ax.set_ylabel(ax_name + "_pred")
        return f, axs


    def plot_box_R0(self, R0, data=None, fns=[], n_box=0.1, ylbl=1, axs=None):
        if axs is None: 
            nPlot = self.nOdx if self.nOdx != 2 else 1
            f, axs = plt.subplots(1, nPlot, figsize=(5*nPlot, 4), facecolor="w")
            if self.nOdx == 2: axs = [axs]
        fns = fns + [self.make_box_fn_R0(R0, n_box=n_box)]
        if data is not None:
            fns = fns +  [BasePlot.scatter_fn(data, c="b")]
        for i, ax in enumerate(axs):
            j = 0 if i == self.nOdx-1 else i + 1
            handles, labels = ax.get_legend_handles_labels()
            handles = []
            for fn in fns:
                handles = fn(i, j, ax, handles)
    
            ax.legend(handles = handles)
            ax.set_xlabel(self.PhyLong[i])            
            # ax.annotate(f"{self.dR[R0]}-NN", xy=(0.5,0.8), xycoords="axes fraction",fontsize=15, c=self.dRC[R0])           
            # if Ps is not None: ax.set_title(f"[M/H] = {Ps[0]:.2f}, Teff={int(Ps[1])}K, logg={Ps[2]:.2f}")
            if ylbl: ax.set_ylabel(self.PhyLong[j])
            if self.nOdx == 2: return f
        return f

    
    def make_box_fn_R0(self, R0, n_box=None, c="k"):
        box_label = f"{Util.DRR[R0]}-box"
        pMin = self.pMins[R0]
        pMax = self.pMaxs[R0]
        pRng = self.pRngs[R0]
        def fn(i, j , ax, handles=[]):
            if n_box is not None:
                ax.set_xlim(pMin[i] - n_box * pRng[i], pMax[i] + n_box * pRng[i])
                ax.set_ylim(pMin[j] - n_box * pRng[j], pMax[j] + n_box * pRng[j])
            ax.add_patch(Rectangle((pMin[i],pMin[j]),(pRng[i]),(pRng[j]),edgecolor=c,lw=2, facecolor="none"))
            handles.append(Patch(facecolor='none', edgecolor=c, label=box_label)) 
            return handles
        return fn

