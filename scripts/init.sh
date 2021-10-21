#!/bin/bash

source ~/.bashrc

# Python environment

source /datascope/slurm/miniconda3/bin/activate ae-tf

############################# SSH AGENT ##################

function is_ssh_agent_running() {
  pgrep -x ssh-agent -u $UID > /dev/null
}

function check_ssh_key() {
  ssh-add -l | grep $1 > /dev/null
}

SSH_KEY=$HOME/.ssh/id_rsa
SSH_ENV=$HOME/.ssh/environment-$HOSTNAME

# Start ssh-agent if it's not already running
if is_ssh_agent_running; then
  echo "ssh-agent already running"
else
  echo "Starting ssh-agent..."
  ssh-agent -s | sed 's/^echo/#echo/' > ${SSH_ENV}
  chmod 600 ${SSH_ENV}
fi

source ${SSH_ENV} > /dev/null

# Add ssh-key to agent
if check_ssh_key $SSH_KEY; then
  echo "Github ssh-key is already added."
else
  echo "Adding Github ssh-key, please enter passphrase."
  ssh-add $SSH_KEY
fi

##########################################################

# PFS related settings
export PYTHONPATH=.
export AE_ROOT=~/AE
# export PFSSPEC_DATA=/scratch/ceph/dobos/data/pfsspec
# export PFSETC=/home/dobos/project/spt_ExposureTimeCalculator

# Work around issues with saving weights when running on multiple threads
export HDF5_USE_FILE_LOCKING=FALSE

# Disable tensorflow deprecation warnings
export TF_CPP_MIN_LOG_LEVEL=3

# Enable more cores for numexpr (for single process operation only!)
# export NUMEXPR_MAX_THREADS=32

# Limit number of threads (for multiprocess computation only!)
# export NUMEXPR_MAX_THREADS=12
# export OMP_NUM_THREADS=12


cd $AE_ROOT

echo "Configured environment for PFS development."
# echo "Data directory is $AE_DATA"

pushd .

