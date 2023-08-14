"""
Here I use Wav2Vec2Model.preprocessed audio array as input audio feature
return all vertices with given semantic part name, and semantic part dict
and with headpose, and shoulder info (i.e. torso)

To make it suit for HDTF_LSP, I update the read dataset logic for convient reading from dir.


@author: DreamTale
@date:   2022/08/25
"""

import sys
from tqdm import tqdm
sys.path.append("..")

from datasets.base_dataset import BaseDataset
# from base_dataset import BaseDataset
import torch
import bisect
import librosa
import os
import mediapipe as mp
import numpy as np
import torch.nn.functional as F
# from models.wav2vec import Wav2Vec2Model
from transformers import Wav2Vec2Processor

# torch.multiprocessing.set_start_method('spawn')

# from funcs import utils


class Audio2ReprDataset(BaseDataset):
    """ audio-visual dataset. currently, return 2D info and 3D tracking info.
        
    """
    def __init__(self, opt):
        super().__init__(opt)
        self.vertices_len = 478
        self.dataroot     = opt.dataroot
        self.sample_rate  = opt.sample_rate
        self.fps          = opt.FPS
        self.max_seq_len  = opt.max_seq_len
        self.use_norm     = opt.use_norm
        self.rand_stretch = opt.rand_stretch
        self.subject_name_lst  = [x.strip() for x in opt.subject_name.split(',')]
        ref_subject_name       = opt.subject_name if opt.ref_sub_names is None else opt.ref_sub_names
        self.ref_sub_name_lst  = [x.strip() for x in ref_subject_name.split(',')]
        self.n_subjects        = len(self.subject_name_lst)
        self.mp_connections    = mp.solutions.face_mesh_connections
        self.frame_jump_stride = opt.frame_jump_stride

        self.semantic_eye_indices = []
        self.semantic_lip_indices = []
        all_semantic_indices = self.get_semantic_indices()
        for k, v in all_semantic_indices.items():
            if 'Left' in k or 'Right' in k:
                self.semantic_eye_indices += v
            elif 'Lip' in k:
                self.semantic_lip_indices += v
        self.eye_vertices_len = len(self.semantic_eye_indices)
        self.lip_vertices_len = len(self.semantic_lip_indices)
        self.vertices_len = 478
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

        self.sample_start = []
        self.total_len    = 0
        self.data_list    = []
        self.subject_len  = [''] * len(self.subject_name_lst)
        for idx, x in enumerate(tqdm(self.subject_name_lst, desc='Loading dataset')):
            subject_data = self.prepare_subject_dataset(self.dataroot, x, self.ref_sub_name_lst[idx])
            self.data_list.append(subject_data)
            t_len = subject_data[2].shape[0] - self.max_seq_len
            self.subject_len[idx] = t_len
            if idx == 0: self.sample_start.append(0)
            else: self.sample_start.append(self.sample_start[-1] + self.subject_len[idx - 1] - 1)
            self.total_len += np.int32(np.floor(t_len / self.frame_jump_stride))

        if hasattr(self, 'processor'):
            del self.processor      # release it 
    
    @staticmethod
    def interp(x, resize_len, mode='bilinear'):
            """
            interplate a list to `resize_len`.
            """
            shape_ori = None
            if len(x.shape) > 1:
                shape_ori = x.shape[1:]
                x = x.reshape(-1, np.prod(shape_ori))
            else:
                x = x[..., None]
            
            x = x[None, None, ...]
            x = F.interpolate(x, size=(resize_len, x.shape[-1]), align_corners=True, mode=mode)
            x = x[0, 0]

            if shape_ori is not None:
                return x.reshape(-1, *shape_ori).float()
            else:
                return x.squeeze()

    @torch.no_grad()
    def prepare_subject_dataset(self, root, data_name, ref_data_name):
        data_dir = os.path.join(root, data_name)
        ref_data_dir = os.path.join(root, ref_data_name)
        facial_info     = np.load(os.path.join(data_dir, 'feature.npz'))
        face_vertices   = facial_info['Vertices']
        face_headposes  = facial_info['Headposes']
        face_transposes = facial_info['Transposes']
        face_scales     = facial_info['Scales'][..., None]
        torso_info      = np.load(os.path.join(data_dir, 'shoulder-billboard.npy'))

        if np.isnan(face_vertices.mean()):
            print(data_dir)
        
        # here reference face vertoces are used to calcute the mean facial mesh as subject info
        facial_info_ref = np.load(os.path.join(ref_data_dir, 'feature.npz'))
        face_vert_ref  = facial_info_ref['Vertices']
        face_head_ref  = facial_info_ref['Headposes']
        face_trans_ref = facial_info_ref['Transposes']
        face_scale_ref = facial_info_ref['Scales'][..., None]
        torso_info_ref = np.load(os.path.join(ref_data_dir, 'shoulder-billboard.npy'))

        audio_array_fp  = os.path.join(data_dir, 'audio_processed.npy')
        if os.path.exists(audio_array_fp):
            audio_array = np.load(audio_array_fp)
        else:
            audio_array, _ = librosa.load(os.path.join(data_dir, 'audio.wav'), sr=None)
            audio_array = np.squeeze(self.processor(audio_array, sampling_rate=16000).input_values)

        def to_tensor(x):
            return torch.from_numpy(x).float()
        
        tmp_audio_len = audio_array.shape[0]
        tmp_video_len = int(tmp_audio_len / self.sample_rate * self.fps)

        return self.interp(to_tensor(audio_array),     tmp_audio_len), int(self.sample_rate / self.fps), \
               self.interp(to_tensor(face_vertices),   tmp_video_len), to_tensor(np.mean(face_vert_ref,  axis=0)), to_tensor(np.std(face_vert_ref,  axis=0)),\
               self.interp(to_tensor(face_headposes),  tmp_video_len), to_tensor(np.mean(face_head_ref,  axis=0)), to_tensor(np.std(face_head_ref,  axis=0)),\
               self.interp(to_tensor(face_transposes), tmp_video_len), to_tensor(np.mean(face_trans_ref, axis=0)), to_tensor(np.std(face_trans_ref, axis=0)),\
               self.interp(to_tensor(face_scales),     tmp_video_len), to_tensor(np.mean(face_scale_ref, axis=0)), to_tensor(np.std(face_scale_ref, axis=0)),\
               self.interp(to_tensor(torso_info),      tmp_video_len), to_tensor(np.mean(torso_info_ref, axis=0)), to_tensor(np.std(torso_info_ref, axis=0)),
    
    def get_semantic_indices(self):

        semantic_connections = {
            'Contours':     self.mp_connections.FACEMESH_CONTOURS,
            'FaceOval':     self.mp_connections.FACEMESH_FACE_OVAL,
            'LeftIris':     self.mp_connections.FACEMESH_LEFT_IRIS,
            'LeftEye':      self.mp_connections.FACEMESH_LEFT_EYE,
            'LeftEyebrow':  self.mp_connections.FACEMESH_LEFT_EYEBROW,
            'RightIris':    self.mp_connections.FACEMESH_RIGHT_IRIS,
            'RightEye':     self.mp_connections.FACEMESH_RIGHT_EYE,
            'RightEyebrow': self.mp_connections.FACEMESH_RIGHT_EYEBROW,
            'Lips':         self.mp_connections.FACEMESH_LIPS,
            'Tesselation':  self.mp_connections.FACEMESH_TESSELATION
        }

        def get_compact_idx(connections):
            ret = []
            for conn in connections:
                ret.append(conn[0])
                ret.append(conn[1])
            
            return sorted(tuple(set(ret)))
        
        semantic_indexes = {k: get_compact_idx(v) for k, v in semantic_connections.items()}

        return semantic_indexes

    def __len__(self):        
        return self.total_len
         
    def zero_padding(self, x, tar_len):
        remain_shape = x.shape[1:]
        ret = torch.zeros((tar_len, *remain_shape))
        ret[:x.shape[0]] = x
        
        return ret.float()

    def __getitem__(self, index):
        # index = 7
        # recover real index from compressed one
        index_real = np.int32(index * self.frame_jump_stride)
        # find which audio file and the start frame index
        file_index = bisect.bisect_right(self.sample_start, index_real) - 1
        current_frame = index_real - self.sample_start[file_index] + int(np.random.randint(0, self.frame_jump_stride//2))
        seq_len = self.max_seq_len

        if self.rand_stretch:
            stretch_ratio = np.random.rand() * 0.5 + 0.75
        else:
            stretch_ratio = 1
        
        seq_len = int(seq_len * stretch_ratio)

        av_rate      = self.data_list[file_index][1]
        audio_arrays = self.data_list[file_index][0][current_frame*av_rate: (current_frame + seq_len)*av_rate] 
        target_mesh  = self.data_list[file_index][2][current_frame:  (current_frame + seq_len)]
        
        target_headpose  = self.data_list[file_index][5][current_frame:  (current_frame + seq_len)]
        target_transpose = self.data_list[file_index][8][current_frame:  (current_frame + seq_len)]
        target_scales    = self.data_list[file_index][11][current_frame: (current_frame + seq_len)]
        target_torso     = self.data_list[file_index][14][current_frame: (current_frame + seq_len)]

        audio_arrays = self.interp(audio_arrays, int(seq_len*av_rate/stretch_ratio))
        target_mesh  = self.interp(target_mesh,  int(seq_len/stretch_ratio))
        target_headpose   = self.interp(target_headpose,  int(seq_len/stretch_ratio))
        target_transpose  = self.interp(target_transpose, int(seq_len/stretch_ratio))
        target_scales = self.interp(target_scales, int(seq_len/stretch_ratio))
        target_torso  = self.interp(target_torso,  int(seq_len/stretch_ratio))

        target_headmotion = torch.cat([target_headpose, target_transpose, target_scales], dim=-1)

        target_headmotion_mean = torch.cat([self.data_list[file_index][6], 
                                            self.data_list[file_index][9],
                                            self.data_list[file_index][12],
                                            ], dim=-1)

        # target_headmotion = (target_headmotion - target_headmotion_mean) / target_headmotion_stdv
        target_headmotion = (target_headmotion - target_headmotion_mean)

        target_mesh_mean = self.data_list[file_index][3]
        # target_mesh = (target_mesh - target_mesh_mean) / target_mesh_stdv
        target_mesh = (target_mesh - target_mesh_mean)

        target_torso_mean = self.data_list[file_index][15]
        target_torso_stdv = self.data_list[file_index][16]
        target_torso = target_torso - target_torso_mean
        torso_mask = torch.mean(target_torso_stdv) > 0.5

        try:        
            target_headmotion = target_headmotion.reshape((target_headmotion.shape[0], -1))
            target_mesh  = target_mesh.reshape((target_mesh.shape[0], -1))
            target_torso = target_torso.reshape((target_torso.shape[0], -1))
        except Exception as e:
            return self.__getitem__(np.random.randint(0, self.total_len))
        
        audio_arrays      = self.zero_padding(audio_arrays,      self.max_seq_len*av_rate)
        target_mesh       = self.zero_padding(target_mesh,       self.max_seq_len)
        target_headmotion = self.zero_padding(target_headmotion, self.max_seq_len)
        target_torso      = self.zero_padding(target_torso,      self.max_seq_len)

        subject_mesh    = target_mesh_mean.reshape((target_mesh_mean.shape[0], -1))
        headmotion_mean = target_headmotion_mean.reshape((target_headmotion_mean.shape[0], -1))
        torso_mean      = target_torso_mean.reshape((target_torso_mean.shape[0], -1))

        return audio_arrays, target_mesh, target_headmotion, target_torso, torso_mask, torso_mean, {'Eye': self.semantic_eye_indices, 'Lip': self.semantic_lip_indices, 'MeanMesh': subject_mesh, 'MeanHeadmotion': headmotion_mean}
    
    def get_meanstd(self, index, use_sub_id=False):
        if not use_sub_id:
            # recover real index from compressed one
            index_real = np.int32(index * self.frame_jump_stride)
            # find which audio file and the start frame index
            file_index = bisect.bisect_right(self.sample_start, index_real) - 1
        else:
            file_index = index
        
        mesh_meanstd      = torch.cat(self.data_list[file_index][3: 4],   dim=-1)
        headpose_meanstd  = torch.cat(self.data_list[file_index][6: 7],   dim=-1)
        transpose_meanstd = torch.cat(self.data_list[file_index][9: 10],  dim=-1)
        scales_meanstd    = torch.cat(self.data_list[file_index][12: 13], dim=-1)

        return mesh_meanstd, headpose_meanstd, transpose_meanstd, scales_meanstd
    

if __name__ == '__main__':
    import cv2
    import imageio
    import subprocess
    import soundfile as sf
    from typing import Mapping
    from easydict import EasyDict
    from rich.progress import track
    from scipy.spatial.transform import Rotation as R

    mp_drawing_styles = mp.solutions.drawing_styles
    mp_connections = mp.solutions.face_mesh_connections


    def frames_to_video(frames, save_pwd, fps=float(60), audio_fp=None):
        os.makedirs(os.path.dirname(save_pwd), exist_ok=True)
        video_tmp_path = save_pwd.replace('.mp4', 'tmp.mp4')
        writer = imageio.get_writer(video_tmp_path, fps=fps)

        for frame in frames:
            writer.append_data(frame[..., ::-1])
        
        writer.close()

        if audio_fp is not None and os.path.exists(audio_fp):
            cmd = 'ffmpeg -i "' + video_tmp_path + '" -i "' + audio_fp + '" -c:a aac -b:a 128k -shortest "' + save_pwd + '" -y'
            subprocess.call(cmd, shell=True) 
            os.remove(video_tmp_path)  # remove the template video


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


    def draw_shoulder_points(img, shoulder_points):
        num = int(shoulder_points.shape[0] / 2)
        for i in range(2):
            for j in range(num - 1):
                pt1 = [int(flt) for flt in shoulder_points[i * num + j]]
                pt2 = [int(flt) for flt in shoulder_points[i * num + j + 1]]
                img = cv2.line(img, tuple(pt1)[:2], tuple(pt2)[:2], (225, 225, 225), 2)  # BGR
        
        return img


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


    data_cfg = EasyDict(dict(
        dataroot = '/cto_studio/vistring/liuyunfei/repo/dataset/HDTF-semantic_mesh/HDTF_LSP',
        sample_rate = 16000,
        FPS = 30,
        frame_jump_stride = 300,
        max_seq_len = 600,
        use_norm = True,
        subject_name  = 'May,McStay,Nadella,Obama,Obama1,Obama21',
        ref_sub_names = 'May,McStay,Nadella,Obama,Obama1,Obama21',
        rand_stretch  = True
    ))

    avd = AudioVisualV5Dataset(data_cfg)

    OUT_DIR = 'test_results_tmp_train'
    os.makedirs(OUT_DIR, exist_ok=True)

    for j in range((len(avd))):
        idx  = np.random.randint(0, len(avd))
        data = avd[idx]

        meta = data[-1]
        mesh_mean       = meta['MeanMesh'].cpu().numpy()
        headmotion_mean = meta['MeanHeadmotion'].squeeze().cpu().numpy()
        torso_mean      = data[-2].cpu().numpy()

        frames = []
        audio_wav, target_mesh, target_headmotion, target_torso = data[:4]
        sf.write(os.path.join(OUT_DIR, f'data-{idx:02}.wav'), audio_wav.squeeze().cpu().numpy(), data_cfg.sample_rate)

        target_mesh       = target_mesh.squeeze().cpu().numpy()
        target_headmotion = target_headmotion.squeeze().cpu().numpy()
        target_torso      = target_torso.squeeze().cpu().numpy()

        seq_len = target_mesh.shape[0]
        for i in track(range(seq_len), description=f'Proc {j:02d} => {idx:04d} ...'):
            mesh     = target_mesh[i]
            torso    = target_torso[i]

            mesh  = mesh.reshape((478, 3)) + mesh_mean
            torso = torso.reshape((18, 3)) + torso_mean
            headpose = target_headmotion[i] + headmotion_mean

            mesh_denorm  = denorm_verticies_via_headpose(mesh,  headpose[:3], headpose[3:6], headpose[6], from_3ddfa=True)
            torso_denorm = denorm_verticies_via_headpose(torso, headpose[:3], headpose[3:6], headpose[6], from_3ddfa=True)

            canvas = np.zeros((512, 512, 3), np.uint8) + 10
            canvas = semantic_meshs2figure(mesh_denorm, canvas)
            canvas = draw_shoulder_points(canvas, torso_denorm)

            frames.append(canvas)
        
        frames_to_video(frames, os.path.join(OUT_DIR, f'data-{idx:02}.mp4'), 30, os.path.join(OUT_DIR, f'data-{idx:02}.wav'))

        if j >= 15: break

        
        

