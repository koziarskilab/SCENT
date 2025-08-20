#!/bin/bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --loop=1 > gpu_memory.log &
NVIDIA_PID=$!

python external/chai/run_chai_multiprocess.py \
    --fasta-rec external/chai/demo.in.rec.fasta \
    --fasta-lig external/chai/demo.in.lig.fasta \
    --template data/receptors/ClpP.pdbqt \
    --out external/chai/demo.out.npy \
    --seed 42 \
    --num_diffn_timesteps 200 \
    --num_diffn_samples 5 \
    --device cuda \
    --use_esm_embeddings False \
    --msa_directory /h/290/stephenzlu/rgfn/data/receptors/metadata \
    --num_threads 2

kill $NVIDIA_PID
echo "Maximum GPU memory used: $(sort -n gpu_memory.log | tail -1) MB"
rm gpu_memory.log
