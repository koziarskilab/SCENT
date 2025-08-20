#!/bin/bash
#SBATCH --job-name=bash          # avoid lightning auto-debug configuration
#SBATCH --output=logs/%x-%j.out  # output file
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1      # shall >#GPU to avoid overtime thread distribution
#SBATCH --cpus-per-task=1        # number of OpenMP threads per MPI process
#SBATCH --mem=96GB
#SBATCH --reservation=mkoziarski_gpu
#SBATCH --gres=gpu:1             # number of GPUs
#SBATCH --time 6-22:59:59        # time limit (D-HH:MM:ss)

#########################
####### Configs #########
#########################
CONDA_ENV_NAME=psalm
CONDA_HOME=$(expr match $CONDA_PREFIX '\(.*miniconda3\)')
WORKDIR=$(pwd)
BASENAME=$(basename $WORKDIR)

#########################
####### Env loader ######
#########################
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_NAME}
# module load miniconda gcc/9.5.0

dt=$(date '+%d/%m/%Y-%H:%M:%S')
echo "[$0] >>> Starttime => ${dt}"

#########################
####### Routine #########
#########################
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# python train.py $@
python train.py $@
