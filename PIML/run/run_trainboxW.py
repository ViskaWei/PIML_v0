from PIML.nn.trainboxW import TrainBoxW

def main():
    d = TrainBoxW()
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
    if test:
        nTrain = 1024
        mtype = "NzDNN"
        train_NL = 2
        nEpoch = 20
        batch = 512
    else:
        nTrain = 131072
        mtype = "NzDNN"
        train_NL = 100
        nEpoch = 5000
        batch = 256

    name=f"B{batch}_T17"

    d.init_train(odx=[0,1,2], mtype=mtype, train_NL=train_NL, nTrain=nTrain, save=1, name=name)
    d.run(lr=0.03, dp=0.02, batch=batch, nEpoch=nEpoch, verbose=1)

if __name__ == "__main__":
    main()