#/bin/bash

if [ -z $1 ]; then
  LOGDIR=$PFSSPEC_DATA/train/run
else
  LOGDIR=$1
fi

tensorboard --logdir=$LOGDIR --host=127.0.0.1 --port=6669 --path_prefix=/tensorboard/`hostname -s`
