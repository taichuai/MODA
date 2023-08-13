import os
import copy
import torch
import librosa
import argparse
import numpy as np
from rich import print
import mediapipe as mp
import albumentations as A
import albumentations.pytorch
from models import create_model
from rich.progress import track
from transformers import Wav2Vec2Processor
from models.componments.wav2vec import Wav2Vec2Model
from options.test_feature_options import TestOptions as FeatureOptions
from utils.util import assign_attributes, parse_config, get_semantic_indices, gauss_smooth_list
from utils.visualizer import frames_to_video, semantic_meshs2figure, get_feature_image, denorm_verticies_via_headpose


mp_drawing_styles = mp.solutions.drawing_styles
mp_connections = mp.solutions.face_mesh_connections


def create_model_by_opt(opt):
    ret = create_model(opt)
    ret.setup(opt)
    ret.load_networks(opt.load_epoch)
    ret.eval()

    return ret


parser = argparse.ArgumentParser()
parser.add_argument('--config',          type=str, default='configs/feature/HDTF.yaml', 
                    help='[!] Override the below parameters from config with cmd args')
parser.add_argument('--meta_dir',       type=str, default='assets/data/meta_dir')
parser.add_argument('--test_person',    type=str, default='Cathy')
parser.add_argument('--force_up',       default=True)
parser.add_argument('--driven_audios',  default='assets/data/test_audios')
parser.add_argument('--out_dir',        type=str, default='results/HDTF-LSP/Render')
parser.add_argument('--a2f_ckpt_fp',    type=str, default='assets/ckpts/MODA.pkl')
parser.add_argument('--refiner_ckpt_fp', type=str, default='assets/ckpts/FaCo.pkl')


args = parser.parse_args()
opts = parse_config(args.config)
assign_attributes(args, opts)


test_opt = FeatureOptions().parse()
test_opt.load_epoch = args.a2f_ckpt_fp
test_opt.extract_wav2vec = True
test_opt.body = 'dualv2'
test_opt.model = 'moda' 
test_opt.subject_head = 'point'

blank_canvas = np.zeros((512, 512, 3), dtype=np.uint8)

faconet_opt = copy.deepcopy(test_opt)
faconet_opt.model = 'faco'
faconet_opt.load_epoch = args.refiner_ckpt_fp
faconet_opt.subject_head = 'point'
n_subjects = 12

all_semantic_indices = get_semantic_indices()

semantic_eye_indices = []
semantic_lip_indices = []
non_semantic_indices = []
for k, v in all_semantic_indices.items():
    if 'Left' in k or 'Right' in k:
        semantic_eye_indices += v
    elif 'Lip' in k:
        semantic_lip_indices += v
    else:
        non_semantic_indices += v

non_semantic_indices = [x for x in non_semantic_indices if x not in semantic_eye_indices and x not in semantic_lip_indices]
semantic_eye_indices = [x for x in semantic_eye_indices if x not in (468, 473)]     # remove L/R iris center points
eye_vertices_len = len(semantic_eye_indices)
lip_vertices_len = len(semantic_lip_indices)

sub_name = args.test_person

test_opt.lip_vertice_dim = lip_vertices_len * 3
test_opt.eye_vertice_dim = eye_vertices_len * 3

MOTPNet = create_model_by_opt(test_opt)
MOTPNet.eval()
FaCoNet = create_model_by_opt(faconet_opt)
FaCoNet.eval()

os.makedirs(args.out_dir, exist_ok=True)
meta = np.load(os.path.join(args.meta_dir, sub_name, 'repr.npz'))

template_mean     = meta['templete']
template_eye_mean = meta['eye']
template_lip_mean = meta['lip']
headmotion_mean   = meta['headmotion']
shoulder_pts_mean = meta['shoulder']

if type(args.driven_audios) is list:
    driven_audios = args.driven_audios
elif os.path.isdir(args.driven_audios):
    driven_audios = [os.path.join(args.driven_audios, x) for x in os.listdir(args.driven_audios)]
else:
    driven_audios = args.driven_audios.split(',')

for driven_audio in driven_audios:

    out_name = f'{"#".join(driven_audio.split(".")[0].split("/")[-2:])}'

    audio_feature_fp = os.path.join(args.out_dir, out_name + '.npy')
    sample_rate = 16000
    fps = 30
    if not os.path.exists(audio_feature_fp):
        wav_path = os.path.join(driven_audio)
        speech_array, _ = librosa.load(os.path.join(wav_path), sr=sample_rate)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        
        audio_feature = np.squeeze(processor(speech_array, sampling_rate=sample_rate).input_values) #[:16000*20]
        audio_feature = np.reshape(audio_feature,(-1, audio_feature.shape[0]))
        np.save(audio_feature_fp, audio_feature)
    else:
        audio_feature = np.load(audio_feature_fp)
    
    if torch.cuda.is_available():
        audio_feature = torch.FloatTensor(audio_feature).cuda()
    else:
        audio_feature = torch.FloatTensor(audio_feature)

    av_rate = sample_rate / fps

    t_out_dir = os.path.join(args.out_dir, sub_name)
    os.makedirs(t_out_dir, exist_ok=True)

    sub_info_aud = template_mean
    sub_info_rfn = template_mean

    print('==== Generate feautre sequence for', out_name, '====')
    seq_fp = os.path.join(t_out_dir, f'seq_pred-{out_name}.npz')
    if not os.path.exists(seq_fp) or args.force_up:
        preds_lip, preds_eye, preds_head, preds_torso = MOTPNet.generate_sequences(audio_feature, n_subjects=n_subjects, sub_id=sub_info_aud, av_rate=av_rate)
        np.savez(seq_fp, lip=preds_lip, eye=preds_eye, hp=preds_head, shoulder=preds_torso)
    else:
        prediction = np.load(seq_fp)
        preds_lip, preds_eye, preds_head, preds_torso = prediction['lip'], prediction['eye'], prediction['hp'], prediction['shoulder']
    
    print('==== Done. ====')

    # reshape vertices 3D
    preds_lip    = np.reshape(preds_lip,    (-1, preds_lip.shape[-1]//3,   3))
    preds_eye    = np.reshape(preds_eye,    (-1, preds_eye.shape[-1]//3,   3))
    preds_torso  = np.reshape(preds_torso,  (-1, preds_torso.shape[-1]//3, 3))

    if opts.with_smooth:
        preds_lip          = gauss_smooth_list(preds_lip,          opts.smooth.lip)
        preds_eye          = gauss_smooth_list(preds_eye,          opts.smooth.refine)
        preds_head[:, :3]  = gauss_smooth_list(preds_head[:, :3],  opts.smooth.head_rot)
        preds_head[:, 3:6] = gauss_smooth_list(preds_head[:, 3:6], opts.smooth.head_trans)
        preds_torso        = gauss_smooth_list(preds_torso,        opts.smooth.torso)
    
    t_pred_lip = torch.from_numpy(preds_lip).to(FaCoNet.device).unsqueeze(0)
    t_pred_eye = torch.from_numpy(preds_eye).to(FaCoNet.device).unsqueeze(0)

    preds_refine = FaCoNet.inference(t_pred_lip, t_pred_eye, n_subjects=n_subjects, sub_id=sub_info_rfn)
    preds_refine = np.reshape(preds_refine, (-1, preds_refine.shape[-1]//3, 3))

    if opts.with_smooth:
        preds_refine[:, non_semantic_indices] = gauss_smooth_list(preds_refine[:, non_semantic_indices], opts.smooth.others)

    frames_lst = []
    for i in track(range(preds_lip.shape[0]), description=f'Processing: {sub_name} => {out_name}, total: {preds_lip.shape[0]}'):

        t_refine_tmp     = preds_refine[i] + template_mean
        t_pred_head     = preds_head[i] + headmotion_mean
        t_pred_shoulder = preds_torso[i] + shoulder_pts_mean
        
        t_pred_refine = denorm_verticies_via_headpose(t_refine_tmp,  t_pred_head[:3], t_pred_head[3:6], t_pred_head[6:])
        t_shoulder   = denorm_verticies_via_headpose(t_pred_shoulder, t_pred_head[:3], t_pred_head[3:6], t_pred_head[6:])
        
        f_viz_pred = get_feature_image(t_pred_refine, (512, 512), t_shoulder, canvas=blank_canvas.copy())  # pred face + pred head + pred shoulder

        frames_lst.append(f_viz_pred)
        

    frames_to_video(frames_lst, os.path.join(t_out_dir, f'{out_name}.mp4'), audio_fp=driven_audio, fps=fps)
    print('Pred results are saved at: ', f'{t_out_dir}/{out_name}.mp4')

