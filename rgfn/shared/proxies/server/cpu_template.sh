#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --output={{LOG_FILE}}
#SBATCH --error={{LOG_FILE}}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time 1:00:00
#SBATCH --reservation=mkoziarski_gpu
# Get the hostname and IP address
HOSTNAME=$(hostname)
IP_ADDR=$(hostname -I | awk '{print $1}')

echo "Starting Proxy server on ${HOSTNAME} (${IP_ADDR}) port {{PORT}}"
echo "Using proxy config: {{PROXY_CONFIG}}"

# Run the Proxy server
python {{SCRIPT_DIR}}/server.py \
    --host ${IP_ADDR} \
    --port {{PORT}} \
    --proxy-config {{PROXY_CONFIG}}
    # # SBATCH --reservation=mkoziarski_gpu
