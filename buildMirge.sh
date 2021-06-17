#!/bin/bash


# default install script for mirgecom, will build with the current development head
echo "Building MIRGE-Com from the current development head using branch pyrometheus-eos"
echo "***WARNING*** may not be compatible with this driver ***WARNING"
echo "consider build scripts from <platforms> as appropriate"

if [ -z "$(ls -A emirge)" ]; then
  git clone git@github.com:illinois-ceesd/emirge.git emirge
else
  echo "emirge install already present. Remove to build anew"
fi

cd emirge

if [ -z ${CONDA_PATH+x} ]; then
  echo "CONDA_PATH unset, installing new conda with emirge"
  ./install.sh --git-ssh --env-name=mirgeDriver.flame1d --git-ssh --branch=y1-production
else
  echo "Using existing Conda installation, ${CONDA_PATH}"
  ./install.sh --git-ssh --conda-prefix=$CONDA_PATH --env-name=mirgeDriver.flame1d --git-ssh --branch=y1-production
fi

# get parsl for remote execution
conda deactivate
conda activate mirgeDriver.flame1d
conda install parsl
