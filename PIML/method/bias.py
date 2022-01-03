
import numpy as np
import matplotlib.pyplot as plt


class Bias():
    def __init__(self, eigv):
        self.eigv = eigv
        self.bias_all = None
        self.bias_1st = None
        self.bias_2nd = None

    def get_bias_from_stdmag(self, stdmag, varmag=None):
        bias_all = self.eigv.dot(np.log(1 + stdmag))
        bias_1st_order = self.eigv.dot(stdmag)
        if varmag is None: varmag = stdmag ** 2
        bias_2nd_order = 1/2 * self.eigv.dot(varmag)
        return bias_all, bias_1st_order, bias_2nd_order

    def plot_theory_bias(self, ak, bias, NL=None, ax=None, pmt=None, log=1, theory=1, N=None, lgd=0, title=None):
        if ax is None: 
            f, ax = plt.subplots(1, figsize=(6,4), facecolor="w")
        b0, b1, b2 = bias
        ak = abs(ak)
        b0 = abs(b0)
        b1 = abs(b1)
        b2 = abs(b2)
        Nname = f"N={N}" if N is not None else ""
        if theory:
            labels =   ["$|\sum_p \log(1+ \sigma_p / A m_p) \cdot V_p|$",  
                        "$|\sum_p (\sigma_p/ A m_p)  \cdot  V_p|$", 
                        "$\sum_p 1/2 \cdot (\sigma_p / A m_p)^2 \cdot V_p$",
                        "Theory $|a_k|$ " + Nname]
        else:
            labels =   [r"$|\sum_p \log(1+ \nu_p / A m_p) \cdot V_p|$",  
                        r"$|\sum_p (\nu_p/ A m_p)  \cdot  V_p|$", 
                        r"$|\sum_p 1/2 \cdot (\nu_p / A m_p)^2 \cdot V_p|$",
                        "$|a_k|$" + Nname]
        ax.plot(ak, b0, 'ko', label=labels[0])
        ax.plot(ak, b1, 'rx', label=labels[1])
        ax.plot(ak, b2, 'bo', label=labels[2])
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel(labels[3])
        # ax.set_xlim(1e-7, 1e-2)
        ax.set_ylim(1e-7, 1e-2)

        if title is not None: ax.set_title(title)
        if lgd: ax.legend(bbox_to_anchor=(0.55, 0.5),  ncol=1)
        # ax.legend()
        return ax

    def plot_exp_bias(self, ak, diffs, labels=None, ax=None, pmt=None, title=None):
        if ax is None: 
            f, ax = plt.subplots(1, figsize=(6,4), facecolor="w")
        ak0 = abs(ak)
        for ii, diff in enumerate(diffs):
            d0 = abs(diff) 
            ax.plot(ak0, d0, 'o', label=labels[ii])
            # ax.plot(ak0, d0 / ak0, 'o', label=labels[ii])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("|$a_k$|")
        ax.set_ylabel("|$b_k$ - $a_k$|")
        # ax.set_ylabel("| $b_k$ / $a_k$ - 1|")
        if title is not None: ax.set_title(title)
        ax.legend()
        return ax