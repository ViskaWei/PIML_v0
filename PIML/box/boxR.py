from PIML.util.basebox import BaseBox


class BoxR(BaseBox):
    def __init__(self):
        super().__init__()


    def init(self, Ws, R, Res, step, topk=10, onPCA=1):
        self.init_Ws(Ws)
        self.init_R(R)
        self.init_plot_R()
        self.Res = Res
        # self.flux, self.pdx0, self.para = self.prepare_data_WR(WRes, R, step, store=True)
        self.run_step_rbf(onPCA, topk)

        self.test_rbf(self.PhyMid, axis=1)
        self.init_LLH()