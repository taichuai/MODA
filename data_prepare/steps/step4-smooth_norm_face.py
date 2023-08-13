"""
Step-4: (Beta) Smooth the meshes

===

Extract audio to wav file from the video file.
Depends:
    rich
    mediapipe
    imageio
"""

__author__ = 'dreamtale'

import os
import cv2
import imageio
import argparse
import numpy as np
from rich import print
import mediapipe as mp
from typing import Mapping
from rich.progress import track
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R



parser = argparse.ArgumentParser(description='Normalize facial vertices')
parser.add_argument('-i', '--in_vertices_fp',  type=str, help='Input vertices file path.', default='temp/temp-norm.npz')
parser.add_argument('-o', '--out_feature_fp',  type=str, help='Output feature file path.', default='temp/temp-smooth.npz')
parser.add_argument('-v', '--out_video_fp',    type=str, help='Output result video file path.', default='temp/temp-smooth.mp4')
parser.add_argument('-ih', '--img_height',     type=int,   default=512)
parser.add_argument('-iw', '--img_width',      type=int,   default=512)

args = parser.parse_args()


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


def smooth_norm_vertices(vertices, sigma=2, exclude_semantics=['LeftIris', 'LeftEye', 'LeftEyebrow', 'RightIris', 'RightEye', 'RightEyebrow', 'Lips']):
    l = vertices.shape[1]
    pts3d = vertices.reshape(-1, l*3)
    pts3d_smooth = gaussian_filter1d(pts3d, sigma, axis=0).reshape(-1, l, 3)
    
    sem_names = get_semantic_indices()
    for es in exclude_semantics:
        pts3d_smooth[:, sem_names[es]] = vertices[:, sem_names[es]]
    
    return pts3d_smooth


features_raw = np.load(args.in_vertices_fp)

other_feats = {k: v for k, v in features_raw.items() if 'Vertices' not in k}
meshes_lst = features_raw['Vertices']
smooth_mesh_list = smooth_norm_vertices(meshes_lst)

writer = imageio.get_writer(args.out_video_fp, fps=30)

for i in track(range(min(meshes_lst.shape[0], smooth_mesh_list.shape[0]))):
    vertices = meshes_lst[i]
    smooth_vertices = smooth_mesh_list[i]

    # relative coord -> abs coord
    vertices[..., 0] *= args.img_width
    vertices[..., 1] *= args.img_height
    vertices[..., 2] *= args.img_width
    smooth_vertices[..., 0] *= args.img_width
    smooth_vertices[..., 1] *= args.img_height
    smooth_vertices[..., 2] *= args.img_width

    blank_canvas = np.zeros((args.img_height, args.img_width, 3), dtype=np.uint8) + 10
    rlt_img = cv2.hconcat([semantic_meshs2figure(vertices, blank_canvas.copy()), 
                           semantic_meshs2figure(smooth_vertices, blank_canvas.copy()),])
    
    writer.append_data(rlt_img[..., ::-1])  # BGR->RGB

writer.close()
np.savez(args.out_feature_fp, 
         Vertices=smooth_mesh_list,
         **other_feats)