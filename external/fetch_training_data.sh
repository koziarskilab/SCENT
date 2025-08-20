#!/bin/bash
# NOTE: our isolation of leaked SMILES uses the same dataset for Chai and Boltz
if [[ $# -eq 0 ]] ; then
    echo "Usage: fetch_leakage.sh <method_name>"
    echo "Valid <method_name>: gnina, chai, boltz, gaa"
    exit 0
fi

METHOD=$1
OUTPUT_DIR=external/training_datasets

# gnina 1.0, ~2.6 GB of space required
GNINA_1_0_DATASETS=(
  "https://bits.csb.pitt.edu/files/gnina1.0_paper/crossdocked_all_data.tar.gz"
  "https://bits.csb.pitt.edu/files/gnina1.0_paper/redocking_all_data.tar.gz"
)

# Boltz-1, ~443 MB of space required
CHAI_BOLTZ_DATASETS=(
  "http://ligand-expo.rcsb.org/dictionaries/Components-pub.cif"
)

# GAABind, ~160 MB of space required
GAA_DATASETS=(
  "http://www.pdbbind.org.cn/download/PDBbind_v2020_sdf.tar.gz"
)

if [[ $METHOD == "gnina" ]]; then
  mkdir -p "$OUTPUT_DIR/gnina"
  cd "$OUTPUT_DIR/gnina"
  wget "${GNINA_1_0_DATASETS[0]}"
  wget "${GNINA_1_0_DATASETS[1]}"
  tar -xf "crossdocked_all_data.tar.gz"
  tar -xf "redocking_all_data.tar.gz"
elif [[ $METHOD == "chai" || $METHOD == "boltz" ]]; then
  mkdir -p "$OUTPUT_DIR/boltz"
  cd "$OUTPUT_DIR/boltz"
  wget "${CHAI_BOLTZ_DATASETS[0]}"
elif [[ $METHOD == "gaa" ]]; then
  mkdir -p "$OUTPUT_DIR/gaa"
  cd "$OUTPUT_DIR/gaa"
  wget "${GAA_DATASETS[0]}"
  tar -xf "PDBbind_v2020_sdf.tar.gz"
fi
