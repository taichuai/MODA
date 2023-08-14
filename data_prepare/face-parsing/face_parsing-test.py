#!/usr/bin/python
# -*- encoding: utf-8 -*-
# This file is a wrapper for face_parsing, will serve as an add-on for step-5

from types import new_class
from logger import setup_logger
from model import BiSeNet

import torch

import os
import sys
import cv2
import imageio
import argparse
import numpy as np
import os.path as osp
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import *
import torchvision.transforms as transforms
import torchvision.transforms.functional as ttf

# Need contain RobustVideoMatting, or clone from here: https://github.com/PeterL1n/RobustVideoMatting.git
DEP_PATH = '~/ws/audio2face/3rdparty/neural-head-avatars/deps'
# You can download it from here: https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth
SEG_MODEL_PATH = "~/ws/audio2face/3rdparty/neural-head-avatars/assets/rvm/rvm_mobilenetv3.pth"
sys.path.append(DEP_PATH)
# include dependency for segmentation
from RobustVideoMatting.model import MattingNetwork


def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    if save_im:
        print(save_path)
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return vis_im


def pad_to_square(img_tensor, mode="replicate"):
        """
        Returns a square (n x n) image by padding the
        shorter edge of
        :param img_tensor: the input image
        :return: squared version of img_tensor, padding information
        """
        y_dim, x_dim = img_tensor.shape[-2:]
        if y_dim < x_dim:
            diff = x_dim - y_dim
            top = diff // 2
            bottom = diff - top
            padding = (0, 0, top, bottom)
        elif x_dim < y_dim:
            diff = y_dim - x_dim
            left = diff // 2
            right = diff - left
            padding = (left, right, 0, 0)
        else:
            return img_tensor, (0, 0, 0, 0)
        return (
            torch.nn.functional.pad(img_tensor[None], padding, mode=mode)[0],
            padding,
        )


def remove_padding(img_tensor, padding):
        """
        Removes padding from input tensor
        :param img_tensor: the input image
        :return: img_tensor without padding
        """
        left, right, top, bottom = padding
        right = -right if right > 0 else None
        bottom = -bottom if bottom > 0 else None

        return img_tensor[..., top:bottom, left:right]


def evaluate(respth='./res/test_res', video_path='data/May-face-centered.avi', cp='model_final_diss.pth', new_size=(512, 512), scale=0.25):

    os.makedirs(respth, exist_ok=True)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    normalize_img = transforms.Compose(
            [
                transforms.Normalize(-1, 2),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    bgr = torch.tensor([0.47, 1, 0.6]).view(3, 1, 1).cuda()  # Green background.
    
    # setting up matting network
    matting_model = MattingNetwork("mobilenetv3").eval().cuda()
    matting_model.load_state_dict(torch.load(SEG_MODEL_PATH))

    rec = [None] * 4  # Set initial recurrent states to None

    cap = cv2.VideoCapture(video_path)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(n_frames)
    base_name = osp.basename(video_path).split('.')[0]

    writer = imageio.get_writer(osp.join(respth, base_name + '-parsing.mp4'), fps=int(cap.get(cv2.CAP_PROP_FPS)))

    parsing_list = []
    downsample_ratio = None
    with torch.no_grad():
        for i in tqdm(range(int(n_frames))):
            ret, frame2 = cap.read()
            if not ret:
                break
            
            image = cv2.resize(frame2, new_size)
            img = ttf.to_tensor(image[..., ::-1].copy()).cuda()[None]
            if downsample_ratio is None:
                downsample_ratio = min(new_size[0] / max(img.shape[-2:]), 1)
            _, pha, *rec = matting_model(img, *rec, downsample_ratio=downsample_ratio)
            
            seg = pha
            # set background to green for better segmentation results
            img = img * seg + bgr * (1 - seg)
            img, padding = pad_to_square(img, mode="constant")
            img = normalize_img(img)
            padded_img_size = img.shape[-2:]
            img = ttf.resize(img, new_size)

            seg_scores = net(img)[0]
            seg_labels = seg_scores.argmax(1, keepdim=True).int()

            # return to original aspect ratio and size
            parsing = seg_labels[0]
            parsing = ttf.resize(parsing, padded_img_size, InterpolationMode.NEAREST)
            parsing = remove_padding(parsing, padding)[0]
            parsing = parsing.cpu().numpy()
            
            h, w = parsing.shape[:2]
            t_parsing = cv2.resize(parsing, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_NEAREST)
            parsing_list.append(t_parsing)
            # print(parsing)
            ret_vis = vis_parsing_maps(image, parsing, stride=1, save_im=False)
            writer.append_data(ret_vis[:, :, ::-1])  # BGR->RGB
    
    writer.close()
    print('Parsing video are saved at  ', osp.join(respth, base_name + '-parsing.mp4'))
    np.save(osp.join(respth, base_name + f'-parsing_{scale}.npy'), np.array(parsing_list))
    print('Parsing result are saved at ', osp.join(respth, base_name + f'-parsing_{scale}.npy'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate shoudler points')
    parser.add_argument('-i', '--video_in_fp', type=str, help='input video path',
                        default='~/ws/audio2face/LiveSpeechPortraits/datasets/tmp/May-face_centered_60fps.avi')
    parser.add_argument('-o', '--output_dir', type=str, help='temp dir for saving intermediate results',
                        default='~/ws/audio2face/LiveSpeechPortraits/dataset/tmp', )
    parser.add_argument('-s', '--scale', type=float, help='re-scale frame size for saving memory',
                        default=0.25, )
    parser.add_argument('-ih', '--img_height',   type=int,   default=512)
    parser.add_argument('-iw', '--img_width',    type=int,   default=512)
    
    args = parser.parse_args()

    img_size = (args.img_height, args.img_width)
    evaluate(video_path=args.video_in_fp, cp='~/ws/audio2face/lsp_dataset_preparation/face-parsing/res/cp/79999_iter.pth', 
            respth=args.output_dir, scale=args.scale, new_size=img_size)


