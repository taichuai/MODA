# coding: utf-8
"""
Step-8: Normalize shoulder points by using headpose

===

Depends:
    rich
    scipy
"""

__author__ = 'dreamtale'

import os
import cv2
import argparse
import numpy as np
from rich.progress import track
from rich import print as rprint
from scipy.spatial.transform import Rotation as R


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


def norm_shoulder_via_headpose(vertices, hp, trans, scale, face_center=np.array((256, 235, 128))):
    rx, ry, rz = hp[0], hp[1], hp[2]
    if args.from_3ddfa:
        rot = R.from_euler('xyz', [ry, rx, rz], degrees=True) 
    else:
        rot = R.from_euler('xyz', [ry, -rx, -rz], degrees=True) 
    
    vertices[:, 2] = -vertices[:, 2]
    
    vertices_center = vertices - trans
    vertices_norm = rot.apply(vertices_center)

    vertices_canonical = vertices_norm * scale

    ret_vertices = vertices_canonical + face_center

    return ret_vertices


def main(args):
    """
    Cut audio & video into two clips with train-val rate
    """

    shoulder_all = np.load(args.shoulder_fp)
    feature_all = np.load(args.feature_fp)

    face_center = np.array((args.img_width//2, int(args.img_height/2.2), args.img_width//4))

    nfames = min(shoulder_all.shape[0], feature_all['Vertices'].shape[0])
    shoulder_pts3d_norm_lst = []
    for i in track(range(nfames)):
        pose  = feature_all['Headposes'][i]
        trans = feature_all['Transposes'][i]
        scale = feature_all['Scales'][i][..., None]
        if shoulder_all[i].shape[-1] == 2:
            # 2D version shoulders, estimate z via faces
            shoulder_pts = shoulder_all[i].reshape((-1, 2))
            pts3d = feature_all['Vertices'][i]

            denormed_pts3d = denorm_verticies_via_headpose(pts3d, pose, trans, scale, face_center=face_center)

            shoulder_pts3d = np.zeros((shoulder_pts.shape[0], 3))
            shoulder_pts3d[:, :2] = shoulder_pts
            shoulder_pts3d[:, 2] = denormed_pts3d[:, 2].mean()
        else:
            shoulder_pts3d = shoulder_all[i]

        shoulder_pts3d_norm = norm_shoulder_via_headpose(shoulder_pts3d, pose, trans, scale, face_center=face_center)
        shoulder_pts3d_norm_lst.append(shoulder_pts3d_norm)
    
    os.makedirs(os.path.dirname(args.output_fp), exist_ok=True)
    np.save(args.output_fp, np.array(shoulder_pts3d_norm_lst))

    rprint(f'Done. Results are generated to {args.output_fp}')



parser = argparse.ArgumentParser(description='Generate dataset for LSP')
parser.add_argument('-f', '--feature_fp', type=str, help='input feature path',
                    default='~/repo/dataset/HDTF-semantic_mesh/HDTF_zhubo/xiexieli/feature.npz')
parser.add_argument('-s', '--shoulder_fp', type=str, help='input shoulder path',
                    default='~/repo/dataset/HDTF-semantic_mesh/HDTF_zhubo/xiexieli/shoulderv2.npy')
parser.add_argument('-o', '--output_fp', type=str, help='output dir',
                    default='~/repo/dataset/HDTF-semantic_mesh/HDTF_zhubo/xiexieli/shoulderv2-billboard.npy', )
parser.add_argument('-m',  '--from_3ddfa',   type=bool,  default=True)
parser.add_argument('-ih', '--img_height',   type=int,   default=512)
parser.add_argument('-iw', '--img_width',    type=int,   default=512)

args = parser.parse_args()
main(args)
