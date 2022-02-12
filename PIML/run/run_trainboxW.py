from PIML.nn.trainboxW import TrainBoxW
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

def main():
    d = TrainBoxW()
    Res = 5000
    W="RedM"
    # Rs=["R"]

    # Rs=["B"]
    # Rs=["M"]
    # Rs=["W"] #2020
    Rs=["C"]
    # Rs=["G"]

    Res=5000
    step=10
    topk=10
    d.init_box(W,Rs, Res,step, topk=topk, onPCA=1, load_eigv=1)
    test = 0
    if test:
        nTrain = 1024
        mtype = "NzDNN"
        trainNL = 2
        nEpoch = 20
        batch = 512
    else:
        nTrain = 131072
        mtype = "NzDNN"
        trainNL = 50
        nEpoch = 500
        batch = 512

    name=f"B{batch}_t{topk}"

    d.init_train(odx=[0,1,2], mtype=mtype, trainNL=trainNL, nTrain=nTrain, save=1, name=name)
    d.run(lr=0.03, dp=0.02, batch=batch, nEpoch=nEpoch, verbose=1)

if __name__ == "__main__":
    main()