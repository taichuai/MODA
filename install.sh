#!/bin/bash

COLOR='\033[0;32m'

echo -e "\n${COLOR}Downloading Data ..."
# wget -O mesh.zip "https://keeper.mpdl.mpg.de/f/f158a430ef754edba5ec/?dl=1"
# unzip mesh.zip -d data/
# mv data/mesh/* data/
# rm -rf data/mesh
# rm -rf mesh.zip
# TODO ...

echo -e "\n${COLOR}Downloading Pretrained model ..."
# TODO ...

echo -e "\n${COLOR}Installing conda env..."
conda env create -f environment.yml

echo -e "\n${COLOR}Installation has finished!"
