#!/usr/bin/python
# -*- encoding: utf-8 -*-
# This file is a wrapper for face_parsing, will serve as an add-on for step-5

from logger import setup_logger
from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm
import imageio
import argparse

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

def evaluate(respth='./res/test_res', video_path='data/May-face-centered.avi', cp='model_final_diss.pth', scale=0.25):

    os.makedirs(respth, exist_ok=True)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    cap = cv2.VideoCapture(video_path)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(n_frames)
    base_name = osp.basename(video_path).split('.')[0]

    writer = imageio.get_writer(osp.join(respth, base_name + '-parsing.mp4'), fps=int(cap.get(cv2.CAP_PROP_FPS)))

    parsing_list = []
    with torch.no_grad():
        for i in tqdm(range(int(n_frames))):
            ret, frame2 = cap.read()
            if not ret:
                break
            image = cv2.resize(frame2, (512, 512))
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
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
                        default='/home/liuyunfei/ws/audio2face/LiveSpeechPortraits/datasets/tmp/May-face_centered_60fps.avi')
    parser.add_argument('-o', '--output_dir', type=str, help='temp dir for saving intermediate results',
                        default='/home/liuyunfei/ws/audio2face/LiveSpeechPortraits/dataset/tmp', )
    parser.add_argument('-s', '--scale', type=float, help='re-scale frame size for saving memory',
                        default=0.25, )
    
    args = parser.parse_args()

    evaluate(video_path=args.video_in_fp, cp='/home/liuyunfei/ws/audio2face/lsp_dataset_preparation/face-parsing/res/cp/79999_iter.pth', 
            respth=args.output_dir, scale=args.scale)


