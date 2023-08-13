"""
Step-3: Get person's headpose

===

Depends:
    3DDFA-V2
"""

__author__ = 'dreamtale'

import argparse
import imageio
import numpy as np
from rich.progress import track
import yaml
from collections import deque
import sys
TDDFA_ROOT = '../3DDFA-V2'
if TDDFA_ROOT not in sys.path: sys.path.append(TDDFA_ROOT)

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
# from utils.render_ctypes import render
from utils.functions import landmarks2figure, get_suffix
from utils.pose import viz_pose2


def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()

    # Given a video path
    reader = imageio.get_reader(args.in_video_fp)

    fps = reader.get_meta_data()['fps']
    duration = reader.get_meta_data()['duration']
    video_wfp = args.out_video_fp
    headpose_wfp = args.out_feature_fp
    writer = imageio.get_writer(video_wfp, fps=fps)

    # the simple implementation of average smoothing by looking ahead by n_next frames
    # assert the frames of the video >= n
    n_pre, n_next = args.n_pre, args.n_next
    n = n_pre + n_next + 1
    queue_ver = deque()
    queue_frame = deque()

    # run
    dense_flag = False
    pre_ver = None
    ver_lst = []
    pose_lst = []
    for i in track(range(int(fps*duration)), description='Processing'):
        try:
            frame_bgr = reader.get_next_data()[..., ::-1]  # RGB->BGR
        except Exception as e: continue

        if i == 0:
            # detect
            boxes = face_boxes(frame_bgr)
            boxes = [boxes[0]]
            param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            # refine
            param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy='landmark')
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            # padding queue
            for _ in range(n_pre):
                queue_ver.append(ver.copy())
            queue_ver.append(ver.copy())

            for _ in range(n_pre):
                queue_frame.append(frame_bgr.copy())
            queue_frame.append(frame_bgr.copy())

        else:
            param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy='landmark')

            roi_box = roi_box_lst[0]
            # todo: add confidence threshold to judge the tracking is failed
            if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                boxes = face_boxes(frame_bgr)
                boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            queue_ver.append(ver.copy())
            queue_frame.append(frame_bgr.copy())

        pre_ver = ver  # for tracking

        # smoothing: enqueue and dequeue ops
        if len(queue_ver) >= n:
            ver_ave_ori = np.mean(queue_ver, axis=0)
            ver_ave = np.transpose(ver_ave_ori, (1, 0))     # (3, 68) => (68, 3)

            ver_lst.append(ver_ave)

            img_draw, poses, cams, scales = viz_pose2(np.uint8(frame_bgr).copy(), param_lst, [ver_ave_ori], show_flag=False, show_info=False)
            pose_lst.append(poses)

            writer.append_data(img_draw[:, :, ::-1])  # BGR->RGB

            queue_ver.popleft()
            queue_frame.popleft()

    # we will lost the last n_next frames, still padding
    for _ in range(n_next):
        queue_ver.append(ver.copy())
        queue_frame.append(frame_bgr.copy())  # the last frame

        ver_ave_ori = np.mean(queue_ver, axis=0)
        ver_ave = np.transpose(ver_ave_ori, (1, 0))     # (3, 68) => (68, 3)

        ver_lst.append(ver_ave)

        img_draw, poses, cams, scales = viz_pose2(np.uint8(frame_bgr).copy(), param_lst, [ver_ave_ori], show_flag=False, show_info=False)
        pose_lst.append(poses)

        writer.append_data(img_draw[..., ::-1])  # BGR->RGB

        queue_ver.popleft()
        queue_frame.popleft()
    
    saved_pose = np.squeeze(np.array(pose_lst))
    print(saved_pose.shape)
    np.save(headpose_wfp, saved_pose)
    print(f'3D features are saved to {headpose_wfp}')

    writer.close()
    print(f'Dump to {video_wfp}')
    sys.path.pop(-1)


parser = argparse.ArgumentParser(description='Extract headpose info')
parser.add_argument('-c', '--config', type=str, 
                        default='/home/liuyunfei/ws/audio2face/lsp_dataset_preparation/3DDFA-V2/configs/mb1_120x120.yml')
parser.add_argument('-i',  '--in_video_fp',    type=str, help='Input video file path.', default='/home/liuyunfei/repo/dataset/HDTF/RD_Radio8_000.mp4')
parser.add_argument('-fi', '--in_feature_fp',  type=str, help='Input feature file path.', default='temp/temp.npz')
parser.add_argument('-fo', '--out_feature_fp', type=str, help='Output feature file path.', default='temp/temp-hp.npy')
parser.add_argument('-v',  '--out_video_fp',   type=str, help='Output result video file path.', default='temp/temp-hp.mp4')
parser.add_argument('-m', '--mode', default='gpu', type=str, help='gpu or cpu mode')
parser.add_argument('-n_pre',  default=1, type=int, help='the pre frames of smoothing')
parser.add_argument('-n_next', default=1, type=int, help='the next frames of smoothing')
parser.add_argument('--onnx', action='store_true', default=True)


args = parser.parse_args()
main(args)
