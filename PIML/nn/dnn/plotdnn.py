import numpy as np
from PIML.util.baseplot import BasePlot
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Patch
from PIML.util.util import Util
import seaborn as sns



class PlotDNN(BasePlot):
    def __init__(self, odx):
        self.odx = odx
        self.nOdx = len(odx)
        self.Rs = None
        self.RRs = None
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
        box_label = f"{Util.DRR[R0]}"
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

    def get_crossover_R0_R1(self, R0, p_pred_R1):
        mask_ij = []
        mask_All = True
        N = len(p_pred_R1)
        for pdx in range(self.nOdx):
            mask0 = True & (p_pred_R1[:,pdx] >= self.pMins[R0][pdx]) & (p_pred_R1[:,pdx] <= self.pMaxs[R0][pdx])
            mask_All = mask_All & mask0
            mask_ij.append(mask0)
        crossover = mask_All.sum() / N

        crossover_ijs = []
        for ii, mask_ii in enumerate(mask_ij):
            jj = ii + 1 if ii < self.nOdx-1 else 0
            mask_jj = mask_ij[jj]
            crossover_ij = (mask_ii & mask_jj).sum() / N 
            crossover_ijs.append(crossover_ij)
        crossover_ijs = np.array(crossover_ijs)
            # print(f"{self.PhyLong[ii]}-{self.PhyLong[jj]} = {crossover_ij:.2f}")
        return crossover, crossover_ijs

    def get_crossMat(self, p_preds, plot=1):
        R0s = p_preds.keys()
        nR0 = len(R0s)
        nR1 = len(self.Rs)
        mat = np.zeros((nR0, nR1))
        mat_ij = np.zeros((nR0, nR1, self.nOdx))

        for ii, (R0, p_preds_R0) in enumerate(p_preds.items()):
            for jj, (R1, p_preds_R0_R1) in enumerate(p_preds_R0.items()):
                # if ii == jj:
                #     CT[ii,jj] = 1 - self.get_contamination_R0_R1(R0, R1)
                # else:
                mat[ii][jj], mat_ij[ii][jj] = self.get_crossover_R0_R1(R0, p_preds_R0_R1)
        
        RR0s = [Util.DRR[R] for R in R0s]
        RR1s = [Util.DRR[R] for R in self.Rs]
        if plot: PlotDNN.plot_heatmap(mat, RR0s, RR1s)
        return mat_ij


    @staticmethod
    def plot_heatmap(mat, RR0s, RR1s=None, vmax=0.5, ax=None):
        if ax is None:
            f, ax = plt.subplots(figsize=(6,5), facecolor="gray")
        sns.heatmap(mat, vmax=vmax, ax=ax, annot=True, cmap="inferno")
        if RR1s is not None: ax.set_xticklabels(RR1s)
        ax.set_yticklabels(RR0s)
        ax.set_title("Crossover Heatmap")