# coding: utf-8

"""
Step-10: Generate feature2image dataset for train/val

As render.
"""

__author__ = 'dreamtale'

import os
import cv2
import shutil
import random
import argparse
import numpy as np
import mediapipe as mp
from typing import Mapping
from rich.progress import track
from rich import print as rprint
from scipy.spatial.transform import Rotation as R


mp_drawing_styles = mp.solutions.drawing_styles
mp_connections = mp.solutions.face_mesh_connections


def draw_landmarks(image, landmarks, connections, landmark_drawing_spec, connection_drawing_spec):
    t_landmarks = np.int32(landmarks[:, :2])
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        drawing_spec = connection_drawing_spec[connection] if isinstance(
            connection_drawing_spec, Mapping) else connection_drawing_spec
        cv2.line(image, t_landmarks[start_idx],
                 t_landmarks[end_idx], drawing_spec.color,
                 drawing_spec.thickness)
    
    if landmark_drawing_spec:
        for idx in range(t_landmarks.shape[0]):
            landmark_px = t_landmarks[idx]
            drawing_spec = landmark_drawing_spec[idx] if isinstance(
            landmark_drawing_spec, Mapping) else landmark_drawing_spec
        # White circle border
        circle_border_radius = max(drawing_spec.circle_radius + 1,
                                    int(drawing_spec.circle_radius * 1.2))
        cv2.circle(image, landmark_px, circle_border_radius, (224, 224, 224),
                    drawing_spec.thickness)
        # Fill color into the circle
        cv2.circle(image, landmark_px, drawing_spec.circle_radius,
                    drawing_spec.color, drawing_spec.thickness)


def semantic_meshs2figure(rlt, canvas):
    ret_img = canvas.copy()
    # draw FACEMESH_TESSELATION
    draw_landmarks(ret_img, rlt, 
        mp.solutions.face_mesh.FACEMESH_TESSELATION, 
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(), 
        landmark_drawing_spec=None)

    # draw FACEMESH_CONTOURS
    draw_landmarks(ret_img, rlt, 
        mp.solutions.face_mesh.FACEMESH_CONTOURS, 
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(), 
        landmark_drawing_spec=None)

    # draw FACEMESH_IRISES
    draw_landmarks(ret_img, rlt, 
        mp.solutions.face_mesh.FACEMESH_IRISES, 
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(), 
        landmark_drawing_spec=None)
    
    return ret_img


def get_semantic_indices():

    semantic_connections = {
        'Contours':     mp_connections.FACEMESH_CONTOURS,
        'FaceOval':     mp_connections.FACEMESH_FACE_OVAL,
        'LeftIris':     mp_connections.FACEMESH_LEFT_IRIS,
        'LeftEye':      mp_connections.FACEMESH_LEFT_EYE,
        'LeftEyebrow':  mp_connections.FACEMESH_LEFT_EYEBROW,
        'RightIris':    mp_connections.FACEMESH_RIGHT_IRIS,
        'RightEye':     mp_connections.FACEMESH_RIGHT_EYE,
        'RightEyebrow': mp_connections.FACEMESH_RIGHT_EYEBROW,
        'Lips':         mp_connections.FACEMESH_LIPS,
        'Tesselation':  mp_connections.FACEMESH_TESSELATION
    }

    def get_compact_idx(connections):
        ret = []
        for conn in connections:
            ret.append(conn[0])
            ret.append(conn[1])
        
        return sorted(tuple(set(ret)))
    
    semantic_indexes = {k: get_compact_idx(v) for k, v in semantic_connections.items()}

    return semantic_indexes


def _draw_shoulder_points(img, shoulder_points):
    if shoulder_points.shape[0] >= 4:
        num = int(shoulder_points.shape[0] / 2)
        for i in range(2):
            for j in range(num - 1):
                pt1 = [int(flt) for flt in shoulder_points[i * num + j]]
                pt2 = [int(flt) for flt in shoulder_points[i * num + j + 1]]
                img = cv2.line(img, tuple(pt1)[:2], tuple(pt2)[:2], (225, 225, 225), 2)  # BGR
    else:
        for j in range(shoulder_points.shape[0] - 1):
            pt1 = [int(flt) for flt in shoulder_points[j]]
            pt2 = [int(flt) for flt in shoulder_points[j + 1]]
            img = cv2.line(img, tuple(pt1)[:2], tuple(pt2)[:2], (225, 225, 225), 2)  # BGR
        
        for i in range(shoulder_points.shape[0]):
            img = cv2.circle(img, np.uint32(shoulder_points[i])[:2], 8, (225, 225, 225), 1)
    
    return img


def _generate_facial_weight_mask(points, img_shape=(512, 512), mouth_outer=None):
    mouth_mask = np.zeros([*img_shape, 1])
    points = points[mouth_outer]
    points = np.int32(points[..., :2])
    mouth_mask = cv2.fillPoly(mouth_mask, [points], (255,0,0))
    mouth_mask = cv2.dilate(mouth_mask, np.ones((45, 45)))
    
    return mouth_mask


def get_feature_image(kps, shoulder_pts, image_shape=(512, 512)):
    canvas = np.zeros((*image_shape, 3), dtype=np.uint8)
    im_edges = semantic_meshs2figure(kps, canvas)  
    im_edges = _draw_shoulder_points(im_edges, shoulder_pts)
    mouth_outer = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
    im_mouth = _generate_facial_weight_mask(kps, image_shape, mouth_outer)

    return im_edges, im_mouth


def denorm_verticies_via_headpose(vertices, hp, eye_center, scale, face_center=np.array((256, 235, 128))):
    vertices_canonical = (vertices - face_center) / scale

    rx, ry, rz = hp[0], hp[1], hp[2]
    if args.from_3ddfa:
        rot = R.from_euler('xyz', [ry, rx, rz], degrees=True).inv()
    else:
        rot = R.from_euler('xyz', [ry, -rx, -rz], degrees=True).inv()

    vertices_center = rot.apply(vertices_canonical)
    ret_ver = vertices_center + eye_center

    return ret_ver


def main(args):
    out_root = args.output_dir
    data_name = 'pix2pix'
    data_dir = os.path.join(out_root, data_name)

    img_tar_dir = os.path.join(data_dir, 'target')
    img_feat_dir = os.path.join(data_dir, 'feature')
    img_fgr_dir = os.path.join(data_dir, 'forground')
    img_mouth_dir = os.path.join(data_dir, 'mouth')

    os.makedirs(img_tar_dir, exist_ok=True)
    os.makedirs(img_feat_dir, exist_ok=True)
    os.makedirs(img_mouth_dir, exist_ok=True)
    os.makedirs(img_fgr_dir, exist_ok=True)
    
    face_center = np.array((args.ref_img_width//2, int(args.ref_img_height/2.2), args.ref_img_width//4))

    ##################### Processing video & features ####################
    rprint("Processing [bold magenta] video [green]...")
    
    cap = cv2.VideoCapture(args.video_fp)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    features_all = np.load(args.feature_fp)
    shouders_all = features_all['Shoulders']

    # Find parsing npy
    parsing_fp = [x for x in os.listdir(os.path.dirname(args.feature_fp)) if 'parsing_0.25' in x][0]
    parsing_all = np.load(os.path.join(os.path.dirname(args.feature_fp), parsing_fp))

    print('n_frames    ', n_frames)
    print('ver_lst',      features_all['Vertices'].shape[0])
    print('shouders_all', shouders_all.shape[0])
    print('parsing_all',  parsing_all.shape[0])

    n_frames = min([n_frames, features_all['Vertices'].shape[0], shouders_all.shape[0]])
    # assert n_frames == features_all['ver_norm_lst'].shape[0] == shouders_all.shape[0]

    for i in track(range(n_frames), description='Processing features'):
        img_feat_fp  = os.path.join(img_feat_dir,  f'{i:05d}.png')
        img_mouth_fp = os.path.join(img_mouth_dir, f'{i:05d}.png') 
        img_fgr_fp   = os.path.join(img_fgr_dir,   f'{i:05d}.png')

        if not os.path.exists(img_feat_fp) or args.force_update:
            kps3d_norm = features_all['Vertices'][i]
            shoulder_pts = shouders_all[i].reshape((-1, 2))

            # recover to abs coord
            kps3d_norm[..., 0] *= args.ref_img_width
            kps3d_norm[..., 1] *= args.ref_img_height
            kps3d_norm[..., 2] *= args.ref_img_width
            shoulder_pts[..., 0] *= args.img_width
            shoulder_pts[..., 1] *= args.img_height

            pose  = features_all['Headposes'][i]
            trans = features_all['Transposes'][i]
            scale = features_all['Scales'][i][..., None]

            kps3d = denorm_verticies_via_headpose(kps3d_norm, pose, trans, scale, face_center=face_center)
            kps3d[..., 0] = kps3d[..., 0] / args.ref_img_width  * args.img_width 
            kps3d[..., 1] = kps3d[..., 1] / args.ref_img_height * args.img_height 
            kps3d[..., 2] = kps3d[..., 2] / args.ref_img_width  * args.img_width 

            img_feat, img_mouth = get_feature_image(kps3d, shoulder_pts, image_shape=(args.img_height, args.img_width))
            
            cv2.imwrite(img_feat_fp,  img_feat)
            cv2.imwrite(img_mouth_fp, img_mouth)
        
        if not os.path.exists(img_fgr_fp) or args.force_update:
            img_fgr = np.uint8(parsing_all[i] > 0) * 255
            img_fgr = cv2.resize(img_fgr, img_mouth.shape[:2][::-1])

            cv2.imwrite(img_fgr_fp,   img_fgr)
    
    for i in track(range(n_frames), description='Processing video'):
        ret, frame = cap.read()
        if ret: 
            img_fp = os.path.join(img_tar_dir, f'{i:05d}.png')
            if not os.path.exists(img_fp): cv2.imwrite(img_fp, frame)
    
    # generate candidate images
    im_names = os.listdir(img_tar_dir)
    im_names_cand = random.choices(im_names, k=4)

    img_cand_dir = os.path.join(data_dir, 'candidate')
    os.makedirs(img_cand_dir, exist_ok=True)
    for name in track(im_names_cand, description='generate candidate'):
        shutil.copy(os.path.join(img_tar_dir, name), img_cand_dir)
        
    with open(os.path.join(out_root, 'train_pix_list.txt'), 'w') as fid:
        for i in range(int(n_frames*args.train_val_rate + 0.5)):
            fid.write(f'{i:05d}\n')

    with open(os.path.join(out_root, 'val_pix_list.txt'), 'w') as fid:
        for i in range(int(n_frames*args.train_val_rate + 0.5), n_frames):
            fid.write(f'{i:05d}\n')

    rprint(f'Done. Results are generated to {out_root}')


parser = argparse.ArgumentParser(description='Generate dataset for Pix2pix')
parser.add_argument('-v', '--video_fp', type=str, help='input video path',
                    default='/home/liuyunfei/repo/dataset/HDTF-semantic_mesh/HDTF_zhubo/xiexieli/video.mp4')
parser.add_argument('-f', '--feature_fp', type=str, help='input feature path',
                    default='/home/liuyunfei/repo/dataset/HDTF-semantic_mesh/HDTF_zhubo/xiexieli/feature.npz')
parser.add_argument('-o', '--output_dir', type=str, help='output dir',
                    default='/home/liuyunfei/repo/dataset/HDTF-semantic_mesh/HDTF_zhubo/xiexieli/renderv2', )
parser.add_argument('--train_val_rate', type=float, default=0.8)
parser.add_argument('--force_update', type=bool, default=False)
parser.add_argument('-m',  '--from_3ddfa',     type=bool,  default=True)
parser.add_argument('-ih', '--img_height',     type=int,   default=512)
parser.add_argument('-iw', '--img_width',      type=int,   default=512)
parser.add_argument('-rh', '--ref_img_height', type=int,   default=768)
parser.add_argument('-rw', '--ref_img_width',  type=int,   default=768)
    
args = parser.parse_args()
main(args)
