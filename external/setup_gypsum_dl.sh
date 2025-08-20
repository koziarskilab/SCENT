#!/bin/bash

export SCRIPT_DIR_NAME=gypsum_dl

wget https://github.com/durrantlab/gypsum_dl/archive/refs/tags/v1.2.1.tar.gz
tar -xzf v1.2.1.tar.gz
mv gypsum_dl-1.2.1 $SCRIPT_DIR_NAME

rm v1.2.1.tar.gz
