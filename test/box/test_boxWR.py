    # def init(self, W, R, Res, step, topk=10, onPCA=1):
    #     self.init_WR(W, R)
    #     self.init_plot_R()
    #     self.Res = Res
    #     self.step = step
    #     self.onPCA = onPCA
    #     self.topk = topk
    #     wave_H, flux_H, self.pdx, self.para = self.IO.load_bosz(Res, RR=self.RR)
    #     self.pdx0 = self.pdx - self.pdx[0]
    #     self.wave_H, flux_H = self.get_flux_in_Wrng(wave_H, flux_H)
    #     # self.flux_H = flux_H
    #     self.Mdx = self.Obs.get_fdx_from_pmt(self.PhyMid, self.para)
    #     self.fluxH0 = flux_H[self.Mdx]

    #     self.wave, self.flux = self.downsample(flux_H)
    #     self.Npix = len(self.wave)
    #     self.flux0 = self.get_model(self.PhyMid, onGrid=1, plot=1)

    #     self.init_sky(self.wave_H, self.flux0, step)

    #     self.logflux = Util.safe_log(self.flux)

    #     self.interp_flux_fn, self.interp_model_fn, self.rbf_sigma, self.interp_bias_fn = self.run_step_rbf(self.logflux, onPCA=onPCA, Obs=self.Obs)
    #     self.init_LLH()
