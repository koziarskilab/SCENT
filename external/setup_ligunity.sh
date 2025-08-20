#!/bin/bash
mkdir external/ligunity
cd external/ligunity
git clone https://github.com/IDEA-XL/LigUnity.git
cd LigUnity/
git checkout a44e1000dd7e2b11944856576a28ddea4a4c0e6f
cd ..

git clone https://github.com/dptech-corp/Uni-Core.git
cd Uni-Core/
git checkout b172ed749b14bf746eea92044b363cd246500f96
python setup.py install
cd ..

git lfs clone https://huggingface.co/fengb/LigUnity_pocket_ranking
