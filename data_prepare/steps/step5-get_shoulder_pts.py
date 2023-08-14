# coding: utf-8
"""
Step-7: Generate shoulder points through face parsing

===

Depends:
    face_parsing
"""

__author__ = 'dreamtale'

import os
import cv2
import sys
import imageio
import argparse
import numpy as np
from collections import deque
from rich.progress import track
from sklearn.cluster import KMeans


FACE_PARSING_ATTRS = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                      'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']


def get_intersection_shoulder(sem:np.ndarray, n_dilation=1):
    # Background class id: 0
    # Cloth class id: 16
    
    t_sem = sem.copy()

    t_sem[np.logical_not(np.logical_or(sem == 0, sem == 16))] = 120

    # calc gradient
    h, w = t_sem.shape[:2]

    grad_x = np.abs(t_sem[1:, :] - t_sem[:-1, :])
    grad_y = np.abs(t_sem[:, 1:] - t_sem[:, :-1])

    grad_x = np.logical_and(15 < grad_x, grad_x < 17)
    grad_y = np.logical_and(15 < grad_y, grad_y < 17)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

    grad_x = cv2.dilate(np.uint8(grad_x)*255, kernel, iterations=n_dilation)
    grad_y = cv2.dilate(np.uint8(grad_y)*255, kernel, iterations=n_dilation)

    grad_x = cv2.resize(grad_x, (w, h), interpolation=cv2.INTER_NEAREST)
    grad_y = cv2.resize(grad_y, (w, h), interpolation=cv2.INTER_NEAREST)

    ret_mask = (np.float32(grad_x) + np.float32(grad_y)) / 2

    return ret_mask > 0


def filter_points(approx_points, n_pts):
    # based on k-means to cluster and get n_pts
    kmeans = KMeans(n_clusters=n_pts, random_state=0).fit(np.squeeze(np.array(approx_points), axis=1))
    ret_pts = []
    for pt_id in range(n_pts):
        points = approx_points[kmeans.labels_==pt_id]
        m_pt = np.mean(np.array(points), axis=0)
        ret_pts.append(m_pt)
    
    ret_pts = sorted(ret_pts, key=lambda x: x[0, 0])
    
    return np.array(ret_pts)


def refine_pts(pts):
    # since the x-axis is stable, we re-sample x-axis and re-compute the y-axis
    n_pts = pts.shape[0]
    assert n_pts > 2

    x_min = pts[:, 0].min()
    x_max = pts[:, 0].max()

    seg_len = (x_max - x_min) / (n_pts - 1)

    cur_id = 0
    ret_pts = []
    for seg_id in range(n_pts):
        new_x = x_min + seg_id * seg_len
        # find the new_x position in origin points
        while cur_id < n_pts and pts[cur_id, 0] < new_x: cur_id += 1
        pt_l = pts[cur_id-1 if cur_id-1 >= 0 else 0]
        pt_r = pts[cur_id]

        rate = np.abs((new_x - pt_l[0]) / max(pt_r[0] - pt_l[0], 1e-4))

        new_y = pt_l[1] * (1 - rate) + pt_r[1] * rate

        ret_pts.append((new_x, new_y))
    
    return np.array(ret_pts)


def fit_shoulder_line_points(shoulder_line, canvas, n_pts):
    t_sl = cv2.resize(np.uint8(shoulder_line)*255, canvas.shape[:2][::-1])

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    t_sl = cv2.dilate(t_sl, kernel, iterations=2)

    contours,_ = cv2.findContours(t_sl, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find the required polygon.
    shoulder_points = []
    h, w = canvas.shape[:2]
    for cnt in contours :
        approx = cv2.approxPolyDP(cnt, 0.003 * cv2.arcLength(cnt, False), False)
        if approx.shape[0] < n_pts:
            continue
        # print(approx)
        # print(approx.shape)
        if (np.max(approx[:, 0, 0], axis=0) - np.min(approx[:, 0, 0], axis=0)) < 50: continue
        if (160/512*w < approx[:, 0, 0].mean() < 352/512*w) and  approx[:, 0, 1].mean() > 400/512*h: continue
        approx_ = filter_points(approx, n_pts) # same shape with approx
        # canvas = cv2.drawContours(canvas, [np.int32(approx_)], 0, (200, 0, 255), 2, lineType=cv2.LINE_AA)
        shoulder_points.append(np.squeeze(np.int32(approx_), axis=1))
        if len(shoulder_points) == 2:
            break
    
    # Safe check, make sure the index-0 is left shouder
    try:
        if shoulder_points[0][0, 0] > shoulder_points[1][0, 0]:
            shoulder_points = shoulder_points[::-1]
    except Exception as e:
        # no shoulders found, fill zero
        shoulder_points = [np.stack([[0, canvas.shape[0]]]   * n_pts, 0), 
                           np.stack([[canvas.shape[0], canvas.shape[0]]] * n_pts, 0)]
    
    # # test code for fixing bad shoulders
    # shoulder_points = [np.stack([[0, canvas.shape[0]]]   * n_pts, 0), 
    #                     np.stack([[canvas.shape[0], canvas.shape[0]]] * n_pts, 0)]

    
    # Here, we refine pts
    shoulder_points = [refine_pts(shoulder_points[0]), refine_pts(shoulder_points[1])]

    return shoulder_points


def draw_sholder_line(canvas, shoulder_line):
    h, w = canvas.shape[:2]
    _sl = cv2.resize(np.uint8(shoulder_line)*255, (w, h))
    show_color = cv2.merge([np.uint8(_sl*0.9), np.uint8(_sl*0.2), np.uint8(_sl*0.1)])
    ret = cv2.addWeighted(canvas, 0.3, show_color, 0.7, 0)
    return ret


def draw_shoulder_points(canvas, shoulder_points, n_pts, colors):
    # visualize
    for t_sz, sps in enumerate(shoulder_points):
        for i in range(len(sps)-1):
            canvas = cv2.line(canvas, sps[i], sps[i+1], colors[i + t_sz*n_pts].tolist(), 3, lineType=cv2.LINE_AA)
        for i in range(len(sps)):
            canvas = cv2.circle(canvas, sps[i], 5, colors[i + t_sz*n_pts].tolist(), 1, lineType=cv2.LINE_AA)
    return canvas


def main(args):
    """
    Face parsing based shoulder keypoints detection and tracking
    """
    file_name = os.path.basename(args.video_fp).split('.')[0]
    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_feat_fp),  exist_ok=True)
    os.makedirs(os.path.dirname(args.output_video_fp), exist_ok=True)

    # Using cmd to run face parsing
    cur_ws = os.getcwd()
    if not os.path.exists(args.face_parsing_dir):
        # Prepare face parsing
        http = 'https://github.com/zllrunning/face-parsing.PyTorch'
        archive = 'https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812'
        root = os.path.dirname(args.face_parsing_dir)
        os.makedirs(root, exist_ok=True)
        os.system(f'git clone {http} {args.face_parsing_dir}')
        os.system('pip install gdown')
        os.system(f"cd {os.path.join(args.face_parsing_dir, 'res', 'cp')}")
        os.system(f"gdown '{archive}'")
        os.system(f'cd {cur_ws}')
    
    os.system(f'cp steps-adv/step5-face_parsing-test.py {args.face_parsing_dir}/face_parsing-test.py')
    sys.path.append(args.face_parsing_dir)
    parsing_fp = os.path.join(args.tmp_dir, file_name + f'-parsing_{args.scale}.npy')
    if not os.path.exists(parsing_fp) or True:
        os.system(f'python {args.face_parsing_dir}/face_parsing-test.py -i {args.video_fp} -o {args.tmp_dir} -s {args.scale} -ih {args.img_height} -iw {args.img_width}')

    # Create some random color table
    colors = np.random.randint(0, 255, (args.n_pts*2, 3))

    parsing_rlt = np.load(parsing_fp)

    n_frames = parsing_rlt.shape[0]

    shoulder_line_lst = [''] * n_frames

    cap = cv2.VideoCapture(os.path.join(args.tmp_dir, file_name + '-parsing.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_shape = (frame_height, frame_width)
    
    assert n_frames == n_count

    out_video = imageio.get_writer(args.output_video_fp, fps=int(fps))

    n_queue = 5
    queue_left  = deque()
    queue_right = deque()

    for idx in track(range(n_frames), description='Processing'):
        sem = parsing_rlt[idx]

        shoulder_line = get_intersection_shoulder(sem)

        # save shoulder line 
        t_sl = cv2.resize(np.uint8(shoulder_line)*255, new_shape, interpolation=cv2.INTER_NEAREST)
        t_sl = cv2.cvtColor(t_sl.copy(), cv2.COLOR_GRAY2BGR)

        frame = np.uint8(np.zeros((*new_shape, 3)) + 255)
        show_img_line = draw_sholder_line(frame, shoulder_line)
        try:
            shou_pts = fit_shoulder_line_points(shoulder_line, show_img_line.copy(), args.n_pts)
        except:
            shou_pts = [queue_left[-1], queue_right[-1]]

        if idx == 0:
            for i in range(n_queue):
                queue_left.append(shou_pts[0])
                queue_right.append(shou_pts[1])
        else:
            # todo: safe check here, filter out the outliers
            if np.mean(np.abs(np.mean(queue_left, axis=0) - shou_pts[0])) < 50:
                queue_left.append(shou_pts[0])
            else:
                queue_left.append(queue_left[-1])
            if np.mean(np.abs(np.mean(queue_right, axis=0) - shou_pts[1])) < 50:
                queue_right.append(shou_pts[1])
            else:
                queue_right.append(queue_right[-1])

        if len(queue_left) >= n_queue and len(queue_right) >= n_queue:
            left_ave = np.int32(np.mean(queue_left, axis=0))
            right_ave = np.int32(np.mean(queue_right, axis=0))

            assert left_ave.shape == right_ave.shape

            shoulder_line_lst[idx] = (left_ave, right_ave)

            show_img_pts = draw_shoulder_points(show_img_line.copy(), (left_ave, right_ave), args.n_pts, colors)
            out_video.append_data(show_img_pts[..., ::-1])
        
            queue_left.popleft()
            queue_right.popleft()

    out_video.close()
    print('Shoulder points video are saved at:', args.output_video_fp)
    features_raw = np.load(args.output_feat_fp)
    other_feats = {k: v for k, v in features_raw.items()}

    shoulders = np.array(shoulder_line_lst, np.float32)
    # to relative coord
    shoulders[:, 0] /= args.img_width
    shoulders[:, 1] /= args.img_height

    np.savez(args.output_feat_fp, 
         Shoulders=shoulders,
         **other_feats)

    print('Shoulder points array is saved at:', args.output_feat_fp, '::[Shoulders]')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate shoudler points')
    parser.add_argument('-v', '--video_fp', type=str, help='input video path',
                        default='~/ws/audio2face/LiveSpeechPortraits/datasets/tmp/May-face_centered_60fps.avi')
    parser.add_argument('-t', '--tmp_dir', type=str, help='temp dir for saving intermediate results',
                        default='~/ws/audio2face/LiveSpeechPortraits/dataset/tmp', )
    parser.add_argument('-of', '--output_feat_fp', type=str, help='output dir',
                        default='~/ws/audio2face/LiveSpeechPortraits/dataset/tmp/shoulder.npy', )
    parser.add_argument('-ov', '--output_video_fp', type=str, help='output dir',
                        default='~/ws/audio2face/LiveSpeechPortraits/dataset/tmp/video.mp4', )
    parser.add_argument('-f', '--face_parsing_dir', type=str, help='face parsing dir',
                        default='face-parsing')
    parser.add_argument('-s', '--scale', type=float, help='re-scale frame size for saving face_parsing memory',
                        default=0.25, )
    parser.add_argument('-n', '--n_pts', type=int, help='number key points on shoulder',
                        default=9, )
    parser.add_argument('-ih', '--img_height',   type=int,   default=512)
    parser.add_argument('-iw', '--img_width',    type=int,   default=512)
    

    args = parser.parse_args()
    main(args)
