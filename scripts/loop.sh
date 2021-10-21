CONFIG=./configs/loop_config.json
LR=(0.003 0.01 0.001)
# LR=(0.001)
DP=(0.001 0.005 0.0)
HD=("1024,512" "512,128,64" "512,128,64,32" "1024,512,256" "128,32")
STD=(0.1 0.01) 
# HD=("10")

for lr in "${LR[@]}"; do
    for hd in "${HD[@]}"; do
        for dp in "${DP[@]}"; do
            for std in "${STD[@]}"; do
                ./scripts/ae.sh \
                    sbatch -p v100 --gpus 1 \
                    --config $CONFIG \
                    --type vae \
                    --stddev $std \
                    --lr $lr \
                    --dropout $dp \
                    --hidden-dims $hd \
                    --epoch 500 \
                    --verbose 0  \
                    --save 1 
            done
        done
    done
done