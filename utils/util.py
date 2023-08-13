import os
import cv2
import yaml
import torch
import subprocess
import numpy as np
import pandas as pd
from PIL import Image
import mediapipe as mp
from rich import print
from typing import Mapping
from easydict import EasyDict
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R


mp_drawing_styles = mp.solutions.drawing_styles
mp_connections = mp.solutions.face_mesh_connections

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if isinstance(image_tensor, torch.autograd.Variable):
        image_tensor = image_tensor.data
    if len(image_tensor.size()) == 5:
        image_tensor = image_tensor[0, -1]
    if len(image_tensor.size()) == 4:
        image_tensor = image_tensor[0]
    image_tensor = image_tensor[:3]
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    #image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * std + mean)  * 255.0        
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)


def add_dummy_to_tensor(tensors, add_size=0):
    if add_size == 0 or tensors is None: return tensors
    if isinstance(tensors, list):
        return [add_dummy_to_tensor(tensor, add_size) for tensor in tensors]    
    
    if isinstance(tensors, torch.Tensor):
        dummy = torch.zeros_like(tensors)[:add_size]
        tensors = torch.cat([dummy, tensors])
    return tensors

def remove_dummy_from_tensor(tensors, remove_size=0):
    if remove_size == 0 or tensors is None: return tensors
    if isinstance(tensors, list):
        return [remove_dummy_from_tensor(tensor, remove_size) for tensor in tensors]    
    
    if isinstance(tensors, torch.Tensor):
        tensors = tensors[remove_size:]
    return tensors

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def parse_config(cfg_fp):
    with open(cfg_fp) as fid:
        cfg = yaml.safe_load(fid)
    return EasyDict(cfg)


def assign_attributes(a_from, a_to):
    for k, v in a_from.items():
        if hasattr(a_to, k):
            setattr(a_to, k, v)


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


def results_to_csv(result_list, out_fp):
    meta = {k:[] for k in result_list[0].keys()}

    # os.makedirs(os.path.dirname(out_fp), exist_ok=True)

    for item in result_list:
        for k in meta.keys():
            meta[k].append(item[k])
    
    for k in meta.keys():
        if k == 'name' or isinstance(meta[k][0], str):
            meta[k].append('Average')
        else:
            meta[k].append(sum(meta[k])/len(meta[k]))

    df = pd.DataFrame(meta)
    df.to_csv(out_fp)


def load_txt(fp):
    with open(fp) as fid:
        datasets = [x.strip() for x in fid.readlines()]
        datasets = [x.split('\t') for x in datasets]

    return datasets


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


def denorm_verticies_via_headpose(vertices, hp, eye_center, scale, face_center=np.array((256, 235, 128)), from_3ddfa=True):
    vertices_canonical = (vertices - face_center) / scale

    rx, ry, rz = hp[0], hp[1], hp[2]
    if from_3ddfa:
        rot = R.from_euler('xyz', [ry, rx, rz], degrees=True).inv()
    else:
        rot = R.from_euler('xyz', [ry, -rx, -rz], degrees=True).inv()

    vertices_center = rot.apply(vertices_canonical)
    ret_ver = vertices_center + eye_center

    return ret_ver


def gauss_smooth_list(in_lst:np.ndarray, smooth_sigma=0):
    """
    in_list: [seq_len, ...]
    """
    ori_shape = np.array(in_lst).shape
    t_a = in_lst.reshape(ori_shape[0], -1)
    t_b = gaussian_filter1d(t_a, smooth_sigma, axis=0)
    
    return t_b.reshape(*ori_shape)


def put_text_on_image(txt, img):
    return cv2.putText(img.copy(), txt, (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (225, 0, 225), 1, cv2.LINE_AA)


def interploate_seq(a_list, tar_seq_len):
    ori_shape = a_list.shape
    t_list = a_list.reshape(ori_shape[0], -1)
    t_list = cv2.resize(t_list, (t_list.shape[-1], tar_seq_len), interpolation=cv2.INTER_NEAREST)
    tar_list = t_list.reshape((tar_seq_len, *ori_shape[1:]))
    return tar_list


def merge_audio_to_video(in_audio_filename, in_video_filename, out_video_filename):
    print('adding audio to video ...')
    cmd = 'ffmpeg -i "' + in_video_filename + '" -i "' + in_audio_filename + '" -c:a aac -b:a 128k -shortest "' + out_video_filename + '" -y -loglevel warning'
    subprocess.call(cmd, shell=True)
    print(f"video with audio was saved to file {out_video_filename}")


def _positional_encoding(tensor, num_encoding_functions=6, include_input=True, log_sampling=True):
    """Apply positional encoding to the input."""
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0**torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0**0.0,
            2.0**(num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=1)


def get_positional_encoding(t_index, position_encoding_size, resolution_inp, num_encoding_functions, include_input):
    t = torch.tensor([t_index]).reshape(1,1,1,1) / position_encoding_size
    t_tens = F.interpolate(t, (resolution_inp, resolution_inp))
    t_embd = _positional_encoding(t_tens, num_encoding_functions=num_encoding_functions, include_input=include_input)
    return t_embd


def prepare_video_data(video_fp, temp_sav_dir):
    video_name = os.path.basename(video_fp).split('.')[0]
    os.makedirs(temp_sav_dir, exist_ok=True)
    
    cmd_extract_frames = f'ffmpeg -i {video_fp} -f image2 {os.path.join(temp_sav_dir, f"{video_name}#frame_%07d.png")} -loglevel warning'
    os.system(cmd_extract_frames)
    audio_fp = os.path.join(temp_sav_dir, f'{video_name}#audio.wav')
    cmd_extract_audio = f'ffmpeg -i {video_fp} -f wav -ar 16000 {audio_fp} -y -loglevel warning'
    os.system(cmd_extract_audio)

    image_pwds = sorted([os.path.join(temp_sav_dir, x) for x in os.listdir(temp_sav_dir) if x.endswith('.png')])
    return image_pwds, audio_fp

