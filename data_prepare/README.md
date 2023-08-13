## Install

1. Build `3DDFA-V2`
    ```shell
    cd 3DDFA-V2
    bash build.sh
    cd ..
    ```
    > If you meet any issues, please visit [here](https://github.com/cleardusk/3DDFA_V2)

2. Prepare `face-parsing`
    ```shell
    git clone https://github.com/zllrunning/face-parsing.PyTorch face-parsing
    pip install gdown
    cd face-parsing/res/cp
    gdown https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812
    cd ../../..
    ```
    > If you meet any issues, please visit [here](https://github.com/zllrunning/face-parsing.PyTorch/issues)
3. Construct your video dataset. It is highly recommended that all videos are with face-centered and **30FPS**.


## Quich run!

```shell
python process.py -i your/video/dir -o your/output/dir
```

Usage:
```shell
usage: process.py [-h] [-i IN_VIDEO_DIR] [-o OUT_DATA_DIR] [-w NUM_WORKERS] [--use_mp USE_MP]

Extract audio info

optional arguments:
  -h, --help            show this help message and exit
  -i IN_VIDEO_DIR, --in_video_dir IN_VIDEO_DIR
                        Input video file dir.
  -o OUT_DATA_DIR, --out_data_dir OUT_DATA_DIR
                        Output dataset dir.
  -w NUM_WORKERS, --num_workers NUM_WORKERS
                        Number of workers for downloading
  --use_mp USE_MP       Whether use multi-processing or not
```

## Gifts!

All scripts in `steps/` are independent running scripts you can use individually.
