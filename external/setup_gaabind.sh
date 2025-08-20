#!/bin/bash

export SCRIPT_DIR_NAME=external/gaabind
# First argument passed is the optional custom environment name (defaults to "gaabind")
export CONDA_ENV_NAME=${1:-gaabind}

# Proceed as in the GAABind repository README
git clone https://github.com/Mercuryhs/GAABind.git $SCRIPT_DIR_NAME
mv external/log_problematic.py $SCRIPT_DIR_NAME
cd $SCRIPT_DIR_NAME
# Modify the environment.yml file so that conda env name matches optional custom name argument
sed -i "1 s/name: .*/name: $CONDA_ENV_NAME/" environment.yml
conda env create -f environment.yml
