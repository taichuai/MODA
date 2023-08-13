# coding: utf-8

__author__ = 'cleardusk'

import numpy as np
import cv2
import collections
from math import sqrt
import matplotlib.pyplot as plt

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)


def get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos:]


def crop_img(img, roi_box):
    h, w = img.shape[:2]

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx
    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
    else:
        res = np.zeros((dh, dw), dtype=np.uint8)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    return res


def calc_hypotenuse(pts):
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    return llength / 3


def parse_roi_box_from_landmark(pts):
    """calc roi box from landmark"""
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]

    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    center_x = (bbox[2] + bbox[0]) / 2
    center_y = (bbox[3] + bbox[1]) / 2

    roi_box = [0] * 4
    roi_box[0] = center_x - llength / 2
    roi_box[1] = center_y - llength / 2
    roi_box[2] = roi_box[0] + llength
    roi_box[3] = roi_box[1] + llength

    return roi_box


def parse_roi_box_from_bbox(bbox):
    left, top, right, bottom = bbox[:4]
    old_size = (right - left + bottom - top) / 2
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0 + old_size * 0.14
    size = int(old_size * 1.58)

    roi_box = [0] * 4
    roi_box[0] = center_x - size / 2
    roi_box[1] = center_y - size / 2
    roi_box[2] = roi_box[0] + size
    roi_box[3] = roi_box[1] + size

    return roi_box


def plot_image(img):
    height, width = img.shape[:2]
    plt.figure(figsize=(12, height / width * 12))

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

    plt.imshow(img[..., ::-1])
    plt.show()


def draw_landmarks(img, pts, style='fancy', wfp=None, show_flag=False, **kwargs):
    """Draw landmarks using matplotlib"""
    height, width = img.shape[:2]
    plt.figure(figsize=(12, height / width * 12))
    plt.imshow(img[..., ::-1])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

    dense_flag = kwargs.get('dense_flag')

    if not type(pts) in [tuple, list]:
        pts = [pts]
    for i in range(len(pts)):
        if dense_flag:
            plt.plot(pts[i][0, ::6], pts[i][1, ::6], 'o', markersize=0.4, color='c', alpha=0.7)
        else:
            alpha = 0.8
            markersize = 4
            lw = 1.5
            color = kwargs.get('color', 'w')
            markeredgecolor = kwargs.get('markeredgecolor', 'black')

            nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]

            # close eyes and mouths
            plot_close = lambda i1, i2: plt.plot([pts[i][0, i1], pts[i][0, i2]], [pts[i][1, i1], pts[i][1, i2]],
                                                 color=color, lw=lw, alpha=alpha - 0.1)
            plot_close(41, 36)
            plot_close(47, 42)
            plot_close(59, 48)
            plot_close(67, 60)

            for ind in range(len(nums) - 1):
                l, r = nums[ind], nums[ind + 1]
                plt.plot(pts[i][0, l:r], pts[i][1, l:r], color=color, lw=lw, alpha=alpha - 0.1)

                plt.plot(pts[i][0, l:r], pts[i][1, l:r], marker='o', linestyle='None', markersize=markersize,
                         color=color, markeredgecolor=markeredgecolor, alpha=alpha)
    if wfp is not None:
        plt.savefig(wfp, dpi=150)
        print(f'Save visualization result to {wfp}')

    if show_flag:
        plt.show()


def cv_draw_landmark(img_ori, pts, box=None, color=GREEN, size=1):
    img = img_ori.copy()
    n = pts.shape[1]
    if n <= 106:
        for i in range(n):
            cv2.circle(img, (int(round(pts[0, i])), int(round(pts[1, i]))), size, color, -1)
    else:
        sep = 1
        for i in range(0, n, sep):
            cv2.circle(img, (int(round(pts[0, i])), int(round(pts[1, i]))), size, color, 1)

    if box is not None:
        left, top, right, bottom = np.round(box).astype(np.int32)
        left_top = (left, top)
        right_top = (right, top)
        right_bottom = (right, bottom)
        left_bottom = (left, bottom)
        cv2.line(img, left_top, right_top, BLUE, 1, cv2.LINE_AA)
        cv2.line(img, right_top, right_bottom, BLUE, 1, cv2.LINE_AA)
        cv2.line(img, right_bottom, left_bottom, BLUE, 1, cv2.LINE_AA)
        cv2.line(img, left_bottom, left_top, BLUE, 1, cv2.LINE_AA)

    return img

pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
pred_types = {'face':       pred_type(slice(0,  17), (0.682, 0.780, 0.909, 1.0)),
              'eyebrow1':   pred_type(slice(17, 22), (1.0,   0.498, 0.055, 0.9)),
              'eyebrow2':   pred_type(slice(22, 27), (1.0,   0.498, 0.055, 0.8)),
              'nose':       pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.9)),
              'nostril':    pred_type(slice(31, 36), (0.545, 0.139, 0.643, 0.9)),
              'eye1':       pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.9)),
              'eye2':       pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.8)),
              'lips_up':    pred_type(slice(48, 55), (0.870, 0.244, 0.210, 0.8)),
              'lips_down':  pred_type(slice(55, 60), (0.930, 0.214, 0.200, 0.8)),
              'teeth_up':   pred_type(slice(60, 65), (0.990, 0.975, 0.899, 0.9)),
              'teeth_down': pred_type(slice(65, 68), (0.990, 0.975, 0.959, 1.0))
              }


def format_cv_colors(x):
    def _c(num):
        return int(num*255)
    return (_c(x[2]), _c(x[1]), _c(x[0]))


def landmarks2figure(landmarks, canvas=None, image_shape=(512, 512), with_line=False):
    """
    landmarks (np.ndarray): 68 3D facial keypoints, shape: (68, 3)
    image_shape (tuple): output image shape. Defaults to (512, 512)
    """
    if canvas is None:
        canvas = np.uint8(np.zeros((*image_shape, 3)) + 223)

    for k, v in pred_types.items():
        kps = landmarks[v.slice, :2]
        # draw line
        if with_line:
            for i in range(len(kps) - 1):
                canvas = cv2.line(canvas, (int(kps[i][0]), int(kps[i][1])), 
                                    (int(kps[i+1][0]), int(kps[i+1][1])), 
                                    format_cv_colors(v.color), 
                                    thickness=2, lineType=cv2.LINE_AA)
            # compensate eye
            if ('eye' in k and 'brow' not in k) or 'teeth' in k or 'lips' in k:
                canvas = cv2.line(canvas, (int(kps[0][0]), int(kps[0][1])), 
                                    (int(kps[-1][0]), int(kps[-1][1])), 
                                    format_cv_colors(v.color), 
                                    thickness=2, lineType=cv2.LINE_AA)
            # draw kps                  
            for i in range(len(kps)):
                canvas = cv2.circle(canvas, (int(kps[i][0]), int(kps[i][1])), 
                                    4, format_cv_colors(v.color), 1, lineType=cv2.LINE_AA)
        else:
            # draw kps
            for i in range(len(kps)):
                canvas = cv2.circle(canvas, (int(kps[i][0]), int(kps[i][1])), 
                                    2, format_cv_colors(v.color), -1, lineType=cv2.LINE_AA)
    
    return canvas


