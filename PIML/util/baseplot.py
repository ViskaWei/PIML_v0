import numpy as np
from scipy.stats import chi2 as chi2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Patch
import matplotlib.transforms as transforms

from matplotlib.lines import Line2D
from matplotlib import collections  as mc

from PIML.util.constants import Constants

class BasePlot(object):
    



# Plot ----------------------------------------------------------------------------------------------------------------------
    def plot_pmt_noise(self, preds, truth, odx, snr, n_box=0.5):
        fns = BasePlot.flow_fn_i(preds, truth, snr, legend=0)
        f = self.plot_box(odx, fns = fns, n_box=n_box)
        f.suptitle(f"SNR = {snr}")
    
    
    @staticmethod
    def get_correlated_dataset(n=1000, cov=[[1, -2],[0.3, 1]], mu=(2,4), scale=(1,10)):
        latent = np.random.randn(n, len(mu))
        dependent = latent.dot(cov)
        scaled = dependent * scale
        scaled_with_offset = scaled + mu
        return scaled_with_offset
    
    def plot3(self, fns=[], data=None, lbl=["MH","Teff","Logg"]):
        f, axs = plt.subplots(1, 3 ,  figsize=(16, 4), facecolor="w")
        for ii, ax in enumerate(axs):
            jj = 0 if ii == 2 else ii + 1
            if data is not None:
                x, y = data[:,jj], data[:,ii]
                ax.scatter(x, y, s=1, alpha=0.5, color="k")
            for fn in fns:
                fn(ii,jj,ax,[])            
            ax.set_xlabel(lbl[ii])            
            ax.set_ylabel(lbl[jj])   

    def plotN(self, fns=[], data=None, N_plot=1, lbl="idx", ann=None, axs=None, outf=False):
        #plot PC coefficients
        if axs is None:
            outf = True
            f, axs = plt.subplots(1,N_plot, figsize=(N_plot*3,3), facecolor="w")
        for ii, ax in enumerate(axs):
            idx1, idx2 = 2 * ii, 2 * ii + 1
            if data is not None:
                x, y = data[:,idx2], data[:,idx1]
                ax.scatter(x, y, s=1, alpha=0.5, color="k")
            for fn in fns:
                fn(idx1,idx2,ax,[])            
            ax.set_xlabel(f"{lbl}{idx1}")            
            ax.set_ylabel(f"{lbl}{idx2}") 
        if outf: return f

    @staticmethod
    def set_unique_legend(ax):
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

    # def flow_fn(self, paras, center, legend=0):
    #     fpara=BasePlot.scatter_fn(paras, c="r",s=10)
    #     fmean=BasePlot.scatter_fn(center, c="g",s=10)
    #     ftraj=BasePlot.traj_fn(paras, center, c="r",lw=2)
    #     return [fpara,fmean, ftraj]

    @staticmethod
    def flow_fn_i(preds, center, snr=None, legend=0):
        mu, sigma = preds.mean(0), preds.std(0)
        center = np.array([center])
        MU = np.array([mu])
        lgd =f"snr={snr}" if legend else None
        
        fpred=BasePlot.scatter_fn(preds, c="gray",s=10, lgd=lgd)
        fmean=BasePlot.scatter_fn(MU,  c="r",s=10)
        ftarget=BasePlot.scatter_fn(center,  c="g",s=10)
        ftraj=BasePlot.traj_fn(MU, center, c="r",lw=2)
        add_ellipse = BasePlot.get_ellipse_fn(preds,c='b', legend=legend)

        return [fpred,fmean, ftarget,ftraj, add_ellipse]

    @staticmethod
    def traj_fn(strts, ends, c=None, lw=2):
        nTest=strts.shape[0]
        def fn(i, j , ax, handles=[]):
            flowList=[]
            for ii in range(nTest):
                strt=strts[ii]
                end= ends[ii]
                flowList.append([(strt[i],strt[j]), (end[i],end[j])])
            lc = mc.LineCollection(flowList, colors=c, linewidths=lw)
            ax.add_collection(lc)
            return handles
        return fn


    @staticmethod
    def scatter_fn(data, c=None, s=1, lgd=None):
        def fn(i, j, ax, handles=[]):
            ax.scatter(data[:,i], data[:,j],s=s, c=c)
            if lgd is not None: 
                handles.append(Line2D([0], [0], marker='o',color='w', label=lgd, markerfacecolor=c, markersize=10))
            return handles
        return fn

    @staticmethod
    def get_ellipse_params(pred, npdx=None):
        if npdx is None: npdx = pred.shape[1]
        x0s,y0s,s05s,degs = [],[],[],[]
        for ii in range(npdx):
            jj = 0 if ii == npdx-1 else ii + 1
            x0, y0, s05, degree = BasePlot.get_ellipse_param(pred[:,ii], pred[:,jj])
            x0s.append(x0)
            y0s.append(y0)
            s05s.append(s05)
            degs.append(degree)
        return x0s,y0s,s05s,degs

    @staticmethod
    def get_ellipse_param(x, y):
        x0,y0=x.mean(0),y.mean(0)
        _, s, v = np.linalg.svd(np.cov(x,y))
        s05 = s**0.5
        degree = BasePlot.get_angle_from_v(v)
        return x0, y0, s05, degree

    @staticmethod
    def get_ellipse_fn(data, c=None, ratio=0.95, legend=1):
        x0s,y0s,s05s,degrees = BasePlot.get_ellipse_params(data)
        chi2_val = chi2.ppf(ratio, 2)
        co = 2 * chi2_val**0.5
        if c is None: c = "r"
        def add_ellipse(i, j, ax, handles):
            x0, y0, s05, degree = x0s[i], y0s[i], s05s[i], degrees[i]
            e = Ellipse(xy=(0,0),width=co*s05[0], height=co*s05[1], facecolor="none",edgecolor=c,)
            transf = transforms.Affine2D().rotate_deg(degree).translate(x0,y0) + ax.transData        
            e.set_transform(transf)
            ax.add_patch(e)
            if legend:
                handles.append(Ellipse(xy=(0,0),width=2, height=1, facecolor="none",edgecolor=c,label=f"Chi2_{100*ratio:.0f}%"))
            return handles
        return add_ellipse

    @staticmethod
    def box_fn(pRange, pMin, pMax,  n_box=None, c="k", RR=None):
        box_label = "box" if RR is None else f"{RR}-box"
        def fn(i, j , ax, handles=[]):
            if n_box is not None:
                ax.set_xlim(pMin[i]-n_box*pRange[i], pMax[i]+n_box*pRange[i])
                ax.set_ylim(pMin[j]-n_box*pRange[j], pMax[j]+n_box*pRange[j])
            ax.add_patch(Rectangle((pMin[i],pMin[j]),(pRange[i]),(pRange[j]),edgecolor=c,lw=2, facecolor="none"))
            handles.append(Patch(facecolor='none', edgecolor=c, label=box_label)) 
            return handles
        return fn



    def plot_box(self, pdxs, data=None, fns=[], n_box=0.1, ylbl=1, axs=None):
        npdx = len(pdxs)
        if axs is None: 
            nPlot = npdx if npdx != 2 else 1
            f, axs = plt.subplots(1, nPlot, figsize=(5*nPlot, 4), facecolor="w")
            if npdx == 2: axs = [axs]
        fns = fns + [self.make_box_fn(n_box)]
        if data is not None:
            fns = fns +  [BasePlot.scatter_fn(data, c="b")]
        for i, ax in enumerate(axs):
            j = 0 if i == npdx-1 else i + 1
            handles, labels = ax.get_legend_handles_labels()
            handles = []
            for fn in fns:
                handles = fn(i, j, ax, handles)
    
            ax.legend(handles = handles)
            ax.set_xlabel(Constants.PhyLong[pdxs[i]])            
            # ax.annotate(f"{self.dR[R0]}-NN", xy=(0.5,0.8), xycoords="axes fraction",fontsize=15, c=self.dRC[R0])           
            # if Ps is not None: ax.set_title(f"[M/H] = {Ps[0]:.2f}, Teff={int(Ps[1])}K, logg={Ps[2]:.2f}")
            if ylbl: ax.set_ylabel(Constants.PhyLong[pdxs[j]])
            if npdx == 2: return f
        return f


    def make_box_fn(self, n_box):
        pass
        
    @staticmethod
    def get_angle_from_v(v, idx=0):
        radian =np.arctan(v[idx][1] / v[idx][0])
        degree = radian / np.pi * 180    
        return degree



    @staticmethod
    def plot_correlated(x,y, chis=[0.95,0.99]):
        fig, ax = plt.subplots(figsize=(10,10))
        ax.scatter(x,y)
        x0,y0=x.mean(0),y.mean(0)
        ax.plot(x0,y0,"ro")
        add_e = BasePlot.get_ellipse_fn_2d(x,y,2)
        for ratio in chis:
            add_e(ratio, ax=ax)
        # ax.set_xlabel("x")
        # ax.set_ylabel("flux")

        # ax.set_title("Correlated data")
        ax.legend()

