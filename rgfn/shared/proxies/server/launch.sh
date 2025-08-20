#!/bin/bash

# Default values
NUM_SERVERS=4
BASE_PORT=5555
SLURM_CONFIG=cpu_template.sh
PROXY_CONFIG=configs/proxies/target/qed.gin
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-servers)
      NUM_SERVERS="$2"
      shift 2
      ;;
    --base-port)
      BASE_PORT="$2"
      shift 2
      ;;
    --proxy-config)
      PROXY_CONFIG="$2"
      shift 2
      ;;
    --slurm-config)
      SLURM_CONFIG="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

echo "Launching $NUM_SERVERS Proxy servers..."
echo "Base port: $BASE_PORT"
echo "Path to proxy config: $PROXY_CONFIG"
echo "SLURM config file: $SLURM_CONFIG"

# Create a directory for server logs
LOGS_DIR="${SCRIPT_DIR}/server_logs"
mkdir -p "$LOGS_DIR"

# Create a registry file to keep track of servers
REGISTRY_FILE="${SCRIPT_DIR}/server_registry.json"
echo "[]" > "$REGISTRY_FILE"

# Function to get hostname from a job ID
get_hostname() {
  local job_id=$1
  squeue --job "$job_id" -o "%N" -h
}

# Launch each server on a separate GPU node
for ((i=0; i<NUM_SERVERS; i++)); do
  PORT=$((BASE_PORT + i))
  JOB_NAME="proxy_server_${i}"
  LOG_FILE="${LOGS_DIR}/${JOB_NAME}.log"

  # Path to the SLURM template script
  SLURM_TEMPLATE="${SCRIPT_DIR}/${SLURM_CONFIG}"

  # Create a temporary SLURM job script by copying the template
  TMP_SCRIPT=$(mktemp)
  cp "$SLURM_TEMPLATE" "$TMP_SCRIPT"

  # Replace placeholders in the template with actual values
  sed -i "s|{{JOB_NAME}}|${JOB_NAME}|g" "$TMP_SCRIPT"
  sed -i "s|{{LOG_FILE}}|${LOG_FILE}|g" "$TMP_SCRIPT"
  sed -i "s|{{PORT}}|${PORT}|g" "$TMP_SCRIPT"
  sed -i "s|{{SCRIPT_DIR}}|${SCRIPT_DIR}|g" "$TMP_SCRIPT"
  sed -i "s|{{PROXY_CONFIG}}|${PROXY_CONFIG}|g" "$TMP_SCRIPT"

  # Submit the job
  echo "Submitting job script: $(cat $TMP_SCRIPT)"
  JOB_ID=$(sbatch "$TMP_SCRIPT" | awk '{print $4}')
  echo "Submitted job ID: $JOB_ID"
  rm "$TMP_SCRIPT"

  echo "Submitted Proxy server $i (Job ID: $JOB_ID)"

  # Wait a moment for the job to start and get the hostname
  sleep 2
  HOSTNAME=$(get_hostname "$JOB_ID")

  # Keep trying to get the hostname for up to 30 seconds
  ATTEMPTS=0
  while [[ -z "$HOSTNAME" || "$HOSTNAME" == "(None)" ]] && [[ $ATTEMPTS -lt 15 ]]; do
    sleep 2
    HOSTNAME=$(get_hostname "$JOB_ID")
    ATTEMPTS=$((ATTEMPTS + 1))
  done

  # Create a simple JSON entry and append it to the registry
  NEW_ENTRY="{\"id\": $i, \"job_id\": \"$JOB_ID\", \"hostname\": \"$HOSTNAME\", \"port\": $PORT, \"proxy_config\": \"$PROXY_CONFIG\", \"slurm_config\": \"$SLURM_CONFIG\", \"status\": \"starting\"}"

  # If registry is empty or just [], create a new array
  if [ ! -s "$REGISTRY_FILE" ] || [ "$(cat "$REGISTRY_FILE")" = "[]" ]; then
    echo "[$NEW_ENTRY]" > "$REGISTRY_FILE"
  else
    # Otherwise, add to the existing array
    sed -i 's/\]$/,/' "$REGISTRY_FILE"
    echo "$NEW_ENTRY]" >> "$REGISTRY_FILE"
  fi

  echo "Server $i running on $HOSTNAME:$PORT (Job ID: $JOB_ID)"
done

echo "All Proxy servers have been launched."
echo "Registry file: $REGISTRY_FILE"
echo "Log directory: $LOGS_DIR"

# Wait for servers to initialize
echo "Waiting for servers to initialize..."
sleep 10

echo "Done. Proxy cluster is now ready."
