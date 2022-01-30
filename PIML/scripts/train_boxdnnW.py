from PIML.nn.dnn.dnnboxW import DnnBoxW

def main():
    d = DnnBoxW()
    Res = 5000
    W="RedM"
    # Rs=["B"]
    Rs=["R"]
    Res=5000
    step=10
    d.init_box(W,Rs, Res,step, topk=10, onPCA=1, load_eigv=1)
    test = 1
    if test:
        nTrain = 1024
        nTest = 100
        mtype = "NzDNN"
        train_NL = 2
        nEpoch = 2
        batch = 512
    else:
        nTrain = 65536
        nTest = 1024
        mtype = "NzDNN"
        train_NL = 10
        nEpoch = 1000
        batch = 16

    d.init_train(out_idx=[0,1,2], mtype=mtype, train_NL=train_NL, nTrain=nTrain, nTest=nTest, log=1)
    d.run(lr=0.01, dp=0.02, batch=batch, nEpoch=nEpoch, verbose=1, eval=0)

if __name__ == "__main__":
    main()