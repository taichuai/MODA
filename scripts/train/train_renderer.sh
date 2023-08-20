conda activate moda

SUB_NAME=$1

python train_renderer.py -c configs/train/renderer/$SUB_NAME.yaml

ln -s xxx assets/ckpts/renderer/$SUB_NAME.pkl
