#!/bin/bash

WORKSPACE=$(dirname $(realpath $0))/qvina
if [ ! -d "$WORKSPACE" ]; then
    mkdir -p "$WORKSPACE"
fi

# Install boost 1.83 from .tar.gz located in external
BOOST_VER=1_83_0
if [ ! -f boost_$BOOST_VER.tar.gz ]; then
    echo "Script running with boost URL unavailable. Please put boost_$BOOST_VER.tar.gz in external. Boost archive not found."
    exit 1
fi
cd "$WORKSPACE"
mv ../boost_$BOOST_VER.tar.gz .
tar -zxvf boost_${BOOST_VER}.tar.gz
cd boost_${BOOST_VER}
./bootstrap.sh --prefix=$WORKSPACE/boost
./b2 install

# Append /boost/lib to LD_LIBRARY_PATH
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$WORKSPACE/boost/lib/" >> ~/.bashrc

# Install QuickVina2-GPU-2.1
cd ..
git clone https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1.git

# Get OPENCL_LIB_PATH for makefile, located at CUDA bin for Nvidia GPU
CUDA_PATH=$(dirname $(dirname $(which nvcc)))

# Compile all 3 versions and modify each Makefile to reflect the correct paths
cd Vina-GPU-2.1
for dir in QuickVina2-GPU-2.1 QuickVina-W-GPU-2.1 AutoDock-Vina-GPU-2.1; do
    cd $dir
    sed -i "s|WORK_DIR=.*|WORK_DIR=$WORKSPACE/Vina-GPU-2.1/$dir|" Makefile
    sed -i "s|BOOST_LIB_PATH=.*|BOOST_LIB_PATH=$WORKSPACE/boost_${BOOST_VER}|" Makefile
    sed -i "s|OPENCL_LIB_PATH=.*|OPENCL_LIB_PATH=$CUDA_PATH|" Makefile
    sed -i "s|-L\$(BOOST_LIB_PATH)/stage/lib|-L$WORKSPACE/boost/lib|" Makefile
    make clean
    make source
    cd ..
done

ln -s "$(dirname $(dirname $(realpath $0)))/data" "QuickVina2-GPU-2.1/data"
