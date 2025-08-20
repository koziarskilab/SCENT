#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --output={{LOG_FILE}}
#SBATCH --error={{LOG_FILE}}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4096
#SBATCH --tmp=32G
#SBATCH --time 6-23:59:59
#SBATCH --gres=gpu:1
#SBATCH --reservation=mkoziarski_gpu
# Get the hostname and IP address
HOSTNAME=$(hostname)
# IP_ADDR=$(hostname -I | awk '{print $1}') # TODO make sure if this line could work. 0.0.0.0 is unsafe.
IP_ADDR=0.0.0.0
echo "Starting Proxy server on ${HOSTNAME} (${IP_ADDR}) port {{PORT}}"
echo "Using proxy config: {{PROXY_CONFIG}}"

# Run the Proxy server
python {{SCRIPT_DIR}}/server.py \
    --host ${IP_ADDR} \
    --port {{PORT}} \
    --proxy-config {{PROXY_CONFIG}}
