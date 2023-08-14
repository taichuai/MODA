"""
Step-3: Normalize face vertices with headpose, transform them to canonical space

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
from scipy.spatial.transform import Rotation as R



parser = argparse.ArgumentParser(description='Normalize facial vertices')
parser.add_argument('-fi', '--in_vertices_fp',  type=str, help='Input vertices file path.',
                    default='~/repo/dataset/HDTF-semantic_mesh/HDTF_LSP/May-test/feature_raw.npz')
parser.add_argument('-fh', '--in_headpose_fp',  type=str, help='Input headpose file path.', 
                    default='~/repo/dataset/HDTF-semantic_mesh/HDTF_LSP/May-test/headpose.npy')
parser.add_argument('-o',  '--out_feature_fp', type=str, help='Output feature file path.', 
                    default='~/repo/dataset/HDTF-semantic_mesh/HDTF_LSP/May-test/feature.npz')
parser.add_argument('-v',  '--out_video_fp',   type=str, help='Output result video file path.', 
                    default='~/repo/dataset/HDTF-semantic_mesh/HDTF_LSP/May-test/norm.mp4')
parser.add_argument('-m',  '--from_3ddfa',   type=bool,  default=True)
parser.add_argument('-ih', '--img_height',   type=int,   default=512)
parser.add_argument('-iw', '--img_width',    type=int,   default=512)

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


def norm_verticies_via_headpose(vertices, hp, face_center=np.array((256, 235, 128))):
    rx, ry, rz = hp[0], hp[1], hp[2]
    if args.from_3ddfa:
        rot = R.from_euler('xyz', [ry, rx, rz], degrees=True) 
    else:
        rot = R.from_euler('xyz', [ry, -rx, -rz], degrees=True) 
    
    # move the two eye's to the zero point 
    sem_ind_lst = get_semantic_indices()
    # eye_center = np.mean((vertices[sem_ind_lst['LeftEye'], :] + vertices[sem_ind_lst['RightEye'], :]) / 2, axis=0)
    # eye_center = vertices[4, :]     # nose
    # eye_center = np.mean(vertices[sem_ind_lst['Contours'], :])
    vertices[:, 2] = -vertices[:, 2]
    eye_center = (vertices[50, :] + vertices[280, :] + vertices[168, :]) / 3    # this version seems more stable than eye center


    vertices_center = vertices - eye_center
    vertices_norm = rot.apply(vertices_center)

    eye_center_dist = np.abs(vertices_norm[sem_ind_lst['LeftEye'], 0] - vertices_norm[sem_ind_lst['RightEye'], 0]).mean()

    scale = 120 / eye_center_dist

    vertices_canonical = vertices_norm * scale

    ret_vertices = vertices_canonical + face_center

    return ret_vertices, eye_center, scale


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


def optimize_symmetry_eyes(vertices):
    # add trick to vert norm for face eye
    left_eye_upper_indices  = [157, 158, 159, 160, 161,  56,  28,  27,  29,  30]
    right_eye_upper_indices = [384, 385, 386, 387, 388, 286, 258, 257, 259, 260]
    y_values = []
    for i in range(len(left_eye_upper_indices)):
        y_values.append(min([vertices[left_eye_upper_indices[i], 1], vertices[right_eye_upper_indices[i], 1]]))
    y_values = np.array(y_values)
    vertices[left_eye_upper_indices, 1]  = 0.4 * vertices[left_eye_upper_indices, 1] + 0.6 * y_values
    vertices[right_eye_upper_indices, 1] = 0.4 * vertices[right_eye_upper_indices, 1] + 0.6 * y_values
    return vertices


features_raw = np.load(args.in_vertices_fp)
headpose_raw = np.load(args.in_headpose_fp)

other_feats = {k: v for k, v in features_raw.items() if 'Vertices' not in k}
meshes_lst = features_raw['Vertices']

writer = imageio.get_writer(args.out_video_fp, fps=30)

out_vertices = []
headpose_lst = []
transpose_lst = []
scale_lst = []
print(meshes_lst.shape)
print(headpose_raw.shape)
face_center = np.array((args.img_width//2, int(args.img_height/2.2), args.img_width//4))
for i in track(range(min(meshes_lst.shape[0], headpose_raw.shape[0]))):
    vertices = meshes_lst[i]
    # recover it to image size
    vertices[..., 0] *= args.img_width
    vertices[..., 1] *= args.img_height
    vertices[..., 2] *= args.img_width

    headpose = headpose_raw[i]

    vert_norm, trans, s = norm_verticies_via_headpose(vertices, headpose, face_center=face_center)
    # vert_norm = optimize_symmetry_eyes(vert_norm)

    headpose_lst.append(headpose)
    transpose_lst.append(trans)
    scale_lst.append(s)
    
    vert_denorm = denorm_verticies_via_headpose(vert_norm, headpose, trans, s, face_center=face_center)

    blank_canvas = np.zeros((args.img_height, args.img_width, 3), dtype=np.uint8) * 100
    rlt_img = cv2.hconcat([semantic_meshs2figure(vertices, blank_canvas.copy()), 
                           semantic_meshs2figure(vert_norm, blank_canvas.copy()),
                           semantic_meshs2figure(vert_denorm, blank_canvas.copy())])
    
    writer.append_data(rlt_img[..., ::-1])  # BGR->RGB

    # make norm to 0~1
    vert_norm[..., 0] /= args.img_width
    vert_norm[..., 1] /= args.img_height
    vert_norm[..., 2] /= args.img_width
    out_vertices.append(vert_norm)

    # if i > 200: break

writer.close()
np.savez(args.out_feature_fp, 
         Vertices=np.array(out_vertices), 
         Headposes=np.array(headpose_lst),
         Transposes=np.array(transpose_lst),
         Scales=np.array(scale_lst),
         **other_feats)


print(args.out_video_fp)