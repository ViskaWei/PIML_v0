from PIML.nn.dnn.dnnboxW import DnnBoxW

def main():
    d = DnnBoxW()
    Res = 5000
    W="RedM"
    Rs=["R"]

    # Rs=["B"]
    # Rs=["M"]
    # Rs=["W"]
    # Rs=["C"]
    # Rs=["G"]

    Res=5000
    step=10
    d.init_box(W,Rs, Res,step, topk=10, onPCA=1, load_eigv=1)
    test = 0
    name="BN"
    if test:
        nTrain = 1024
        mtype = "NzDNN"
        train_NL = 2
        nEpoch = 10
        batch = 512
    else:
        nTrain = 65536
        mtype = "NzDNN"
        train_NL = 10
        nEpoch = 1000
        batch = 16

    d.init_train(out_idx=[0,1,2], mtype=mtype, train_NL=train_NL, nTrain=nTrain, log=1, name=name)
    d.run(lr=0.03, dp=0.02, batch=batch, nEpoch=nEpoch, verbose=1, eval=0)

if __name__ == "__main__":
    main()