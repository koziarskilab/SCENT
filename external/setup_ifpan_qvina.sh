#!/bin/bash

# Set WORKSPACE variable to external/qvina, create the dir if it doesn't exist
WORKSPACE=$(dirname $(realpath $0))/qvina
if [ ! -d "$WORKSPACE" ]; then
    mkdir -p "$WORKSPACE"
fi

# Copy compiled boost, QuickVina folders from /raid/homes/medzik/fteam/vinagpu
cp -r /raid/homes/medzik/fteam/vinagpu/. $WORKSPACE

# NOTE: Not necessary if running as SLURM job, only for running pipeline locally, look below
# Append /boost/lib from Piotr to LD_LIBRARY_PATH in bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WORKSPACE/boost/lib/" >> ~/.bashrc
# Implement LD_LIBRARY_PATH changes without shell restart
source ~/.bashrc



# SLURM batch file for QuickVina that works for me (does not spit out "conda init before activate" error)
# Modify commented-out conda, python lines to suit your needs (Python 3.9 -> 3.10 change is my own doing)

##!/bin/bash
##SBATCH --qos=normal
##SBATCH --account=modellers
##SBATCH --job-name=binding_gypsum_qvina
##SBATCH --partition=dgx_A100
##SBATCH --gpus=1
##SBATCH --cpus-per-task=8
##SBATCH --mem-per-cpu=16G

##conda create -n <chosen_env_name> python=3.10
#source /raid/soft/miniconda/bin/activate /raid/soft/miniconda/envs/pbab_qvina_test

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path_to_qvina>/boost/lib/

#cd ./BindingAffinityPipeline

#pip install -r requirements.txt
#conda install conda-forge::xorg-libxrender

##python -m run --cfg configs/<chosen_config_file>.gin

#wait
