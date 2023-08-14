"""
Step-1: Extract the semantic facial meshes, including mouth, eye, iris, eyebrow, face edge

===

Extract audio to wav file from the video file.
Depends:
    mediapipe
    imageio
    rich
"""

__author__ = 'dreamtale'

import os
import cv2
import math
import imageio
import argparse
import numpy as np
from rich import print
import mediapipe as mp
from typing import Mapping
from collections import deque
from rich.progress import track


parser = argparse.ArgumentParser(description='Extract semantic facial mesh info')
parser.add_argument('-i', '--in_video_fp',    type=str, help='Input video file path.', default='~/repo/dataset/HDTF/RD_Radio8_000.mp4')
parser.add_argument('-f', '--out_feature_fp', type=str, help='Output feature file path.', default='temp/temp.npz')
parser.add_argument('-v', '--out_video_fp',   type=str, help='Output result video file path.', default='temp/temp.mp4')
parser.add_argument('-n_pre',  default=1, type=int, help='the pre frames of smoothing')
parser.add_argument('-n_next', default=1, type=int, help='the next frames of smoothing')

args = parser.parse_args()


mp_mesh_det = mp.solutions.face_mesh.FaceMesh(
                        static_image_mode=False,
                        max_num_faces=3,
                        refine_landmarks=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_connections = mp.solutions.face_mesh_connections

def normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, normalized_z: float, image_width: int,
    image_height: int):
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                            math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        return None
    x_px = normalized_x * image_width
    y_px = normalized_y * image_height
    z_px = normalized_z * image_width
    return x_px, y_px, z_px


def get_facial_mesh(image:np.ndarray):
    # image.flags.writeable = False
    image_rgb = image[..., ::-1]
    h, w = image.shape[:2]
    results = mp_mesh_det.process(image_rgb)

    face_landmarks_list = []
    if results.multi_face_landmarks:
        for landmark_list in results.multi_face_landmarks:
            if not landmark_list:
                face_landmarks_list.append(None)
                continue
            idx_to_coordinates = -np.ones((478, 3))     # 468 + 10 eyes
            for idx, landmark in enumerate(landmark_list.landmark):
                if ((landmark.HasField('visibility') and
                    landmark.visibility < 0.5) or
                    (landmark.HasField('presence') and
                    landmark.presence < 0.5)):
                    continue
                landmark_px = normalized_to_pixel_coordinates(landmark.x, landmark.y, landmark.z, w, h)
                idx_to_coordinates[idx] = landmark_px
            face_landmarks_list.append(idx_to_coordinates)
    
    return np.asarray(face_landmarks_list[0]) if len(face_landmarks_list) > 0 else None


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


os.makedirs(os.path.dirname(args.out_feature_fp), exist_ok=True)
os.makedirs(os.path.dirname(args.out_video_fp), exist_ok=True)


reader = imageio.get_reader(args.in_video_fp)

fps = reader.get_meta_data()['fps']
duration = reader.get_meta_data()['duration']
video_wfp = args.out_video_fp
feat3d_wfp = args.out_feature_fp
writer = imageio.get_writer(video_wfp, fps=fps)

# the simple implementation of average smoothing by looking ahead by n_next frames
# assert the frames of the video >= n
n_pre, n_next = args.n_pre, args.n_next
n = n_pre + n_next + 1
queue_ver = deque()
queue_frame = deque()

# run
pre_ver = None
ver_lst = []
for i in track(range(int(fps*duration)), description='Processing'):

    try:
        frame_bgr = reader.get_next_data()[..., ::-1].copy()  # RGB->BGR
        h, w = frame_bgr.shape[:2]
    except Exception as e:
        print(e)
        break

    if i == 0:
        ver = get_facial_mesh(frame_bgr)

        # padding queue
        for _ in range(n_pre):
            queue_ver.append(ver.copy())
        queue_ver.append(ver.copy())

        for _ in range(n_pre):
            queue_frame.append(frame_bgr.copy())
        queue_frame.append(frame_bgr.copy())

    else:
        ver = get_facial_mesh(frame_bgr)

        if ver is None: ver = queue_ver[-1]
        if np.isnan(ver.mean()):
            ver[np.where(np.isnan(ver))] = queue_ver[-1][np.where(np.isnan(ver))]

        queue_ver.append(ver.copy())
        queue_frame.append(frame_bgr.copy())

    pre_ver = ver  # for tracking

    # smoothing: enqueue and dequeue ops
    if len(queue_ver) >= n:
        ver_ave = np.mean(queue_ver, axis=0)

        ver_lst.append(ver_ave)
        img_draw = semantic_meshs2figure(ver_ave, queue_frame[n_pre].copy())  # since we use padding

        writer.append_data(img_draw[:, :, ::-1])  # BGR->RGB

        queue_ver.popleft()
        queue_frame.popleft()

# we will lost the last n_next frames, still padding
for _ in range(n_next):
    queue_ver.append(ver.copy())
    queue_frame.append(frame_bgr.copy())  # the last frame

    ver_ave = np.mean(queue_ver, axis=0)
    ver_lst.append(ver_ave)

    img_draw = semantic_meshs2figure(ver_ave, queue_frame[n_pre].copy())  # since we use padding

    writer.append_data(img_draw[..., ::-1])  # BGR->RGB

    queue_ver.popleft()
    queue_frame.popleft()

sem_ver_indices = get_semantic_indices()

# norm vers to 0~1
vers = np.array(ver_lst)
vers[..., 0] /= w
vers[..., 1] /= h
vers[..., 2] /= w

np.savez(feat3d_wfp, Vertices=vers, **sem_ver_indices)
print(f'Semantic facial keypoints are saved to {feat3d_wfp}')

writer.close()
print(f'Results are dumpped to {video_wfp}')