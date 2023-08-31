#!/bin/bash

COLOR='\033[0;32m'

echo -e "\n${COLOR}Installing conda env..."
conda env create -f environment.yml
conda activate moda

echo -e "\n${COLOR}Downloading Data ..."

mkdir -p assets/data/test_audios
mkdir -p assets/data/meta_dir/Cathy
gdown --id 1ISMHAoEKIo1_BPGDuWeBWWhz8D9jZwqf --output assets/data/meta_dir/Cathy
gdown --id 1maUQxgVKJakAu4UMkj_T4eYnzaAMktKW --output assets/data/meta_dir/Cathy
gdown --id 1_qytlBAy1VzpUIrSakgUCWRm2dbovEwD --output assets/data/test_audios
gdown --id 1pfcmOzBE9z4GIPI__mhKOGwfj5tdmRuh --output assets/data/test_audios


echo -e "\n${COLOR}Downloading Pretrained model ..."
pip install gdown
mkdir -p assets/ckpts/renderer
gdown --id 1TgyVM1JwKmt1uMkS5nAb9SAKMVCcUZmW --output assets/ckpts
gdown --id 1g1YOQaOT9gHXNpLATGxqX2ysSaaIShRT --output assets/ckpts
gdown --id 14TfL3Qq1TrT4FkFsQOmcBVwZ_Fj-_AjZ --output assets/ckpts/renderer

echo -e "\n${COLOR}Installation has finished!"
