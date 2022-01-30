#/bin/bash

if [ -z $1 ]; then
  LOGDIR=/home/swei20/PhysicsInformedML/logs
else
  LOGDIR=$1
fi

tensorboard --logdir=$LOGDIR --host=127.0.0.1 --port=8711 --path_prefix=/tensorboard/`hostname -s`
