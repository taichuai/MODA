"""
Process the video file to semantic features for taking face dataset

===
Depends:
    see steps-adv
"""

__author__ = 'dreamtale'

import os
import argparse
from tqdm import tqdm
from rich import print
from multiprocessing import Pool
import subprocess

parser = argparse.ArgumentParser(description='Extract audio info')
parser.add_argument('-i', '--in_video_dir', type=str, help='Input video file dir.',
                    default='/cto_studio/vistring/liuyunfei/repo/dataset/HDTF/')
parser.add_argument('-o', '--out_data_dir', type=str, help='Output dataset dir.',
                    default='/home/liuyunfei/repo/dataset/HDTF-processed')
parser.add_argument('-w', '--num_workers', type=int, default=8, 
                    help='Number of workers for downloading')
parser.add_argument('--use_mp', type=bool, default=True, 
                    help='Whether use multi-processing or not')

args = parser.parse_args()

out_dir = args.out_data_dir
os.makedirs(out_dir, exist_ok=True)

def run_cmd(cmd):
    print('RUN => ', cmd)
    subprocess.call(cmd, shell=True)

if os.path.isdir(args.in_video_dir):
    video_names = [x for x in os.listdir(args.in_video_dir) if x.endswith('.mp4') and 'fps' not in x]
else:
    assert os.path.exists(args.in_video_dir)
    video_names = [args.in_video_dir]
# video_names = ['WDA_DonnaShalala1_000.mp4', 'WDA_JanSchakowsky1_000.mp4', 'WDA_RaulRuiz_000.mp4', 'WDA_JohnLewis1_000.mp4']

video_names = [x for x in video_names if 'Obama' not in x and 'McStay' in x]
video_names = sorted(video_names)

force_update = True
intermediate_img_size = 768        # for more accurate features
target_img_size = 512


def proc_video_by_name(idx, vn):
    
    prefix = f'Processing {idx:3d} / {len(video_names)}: {vn}'
    t_outdir = os.path.join(out_dir, vn.split('.')[0])
    os.makedirs(t_outdir, exist_ok=True)

    in_video_fp = os.path.join(args.in_video_dir, vn)
    out_audio_fp = os.path.join(t_outdir, 'audio.wav')
    out_video_fp = os.path.join(t_outdir, 'video.mp4')

    print(f'### {prefix} => Step-1/7: ', 'Extract audio file ...')
    if not os.path.exists(out_audio_fp) or not os.path.exists(out_video_fp) or force_update:
        run_cmd(f'python steps/step0-extract_video_audio.py -i {in_video_fp} -a {out_audio_fp} -v {out_video_fp} --target_h {intermediate_img_size} --target_w {intermediate_img_size}')

    in_video_fp = out_video_fp

    print(f'### {prefix} => Step-2/7: ', 'Extract semantic mesh ...')
    out_video_feat_fp = os.path.join(t_outdir, 'semantic_mesh.mp4')
    out_feat_fp = os.path.join(t_outdir, 'feature_raw.npz')
    if not os.path.exists(out_feat_fp) or force_update:
        run_cmd(f'python steps/step1-get_smooth_semantic_kps.py -i {in_video_fp} -f {out_feat_fp} -v {out_video_feat_fp}')

    print(f'### {prefix} => Step-3/7: ', 'Extract head pose ...')
    out_video_hp_fp = os.path.join(t_outdir, 'headpose.mp4')
    out_feat_hp_fp = os.path.join(t_outdir, 'headpose.npy')
    if not os.path.exists(out_feat_hp_fp) or force_update:
        run_cmd(f'python steps/step2-get_smooth_headpose-tddfa.py -i {in_video_fp} -fi {out_feat_fp} -fo {out_feat_hp_fp} -v {out_video_hp_fp}')

    print(f'### {prefix} => Step-4/7: ', 'Computing canonical mesh ...')
    out_video_norm_fp = os.path.join(t_outdir, 'norm.mp4')
    out_feat_norm_fp = os.path.join(t_outdir, 'feature.npz')
    if not os.path.exists(out_feat_norm_fp) or force_update:
        run_cmd(f'python steps/step3-norm_face.py -fi {out_feat_fp} -fh {out_feat_hp_fp} -o {out_feat_norm_fp} -v {out_video_norm_fp} -m {1} -ih {intermediate_img_size} -iw {intermediate_img_size}')

    print(f'### {prefix} => Step-5/7: ', 'Smooth canonical mesh ...')
    out_video_smooth_fp = os.path.join(t_outdir, 'norm-smooth.mp4')
    if not os.path.exists(out_video_smooth_fp) or force_update:
        run_cmd(f'python steps/step4-smooth_norm_face.py -i {out_feat_fp} -o {out_feat_fp} -v {out_video_smooth_fp} -ih {intermediate_img_size} -iw {intermediate_img_size}')

    print(f'### {prefix} => Step-6/7: ', 'Computing shoulder points ...')
    out_video_shoulder_fp = os.path.join(t_outdir, 'shoulder-refine_adv.mp4')
    out_feature_shoulder_fp = os.path.join(t_outdir, 'shoulder.npy')
    if not os.path.exists(out_feature_shoulder_fp) or force_update:
        run_cmd(f'python steps/step5-get_shoulder_pts.py -v {in_video_fp} -t {t_outdir} -ov {out_video_shoulder_fp} -of {out_feat_norm_fp}  -ih {intermediate_img_size} -iw {intermediate_img_size}')
    
    print(f'### {prefix} => Step-7/7: ', 'Generate render dataset ...')
    out_pix_dir = os.path.join(t_outdir, 'render')
    if not os.path.exists(out_pix_dir) or force_update or len(os.listdir(os.path.join(out_pix_dir, 'pix2pix', 'feature'))) < 50:
        run_cmd(f'python steps/step7-gen_pix2pix.py -v {in_video_fp} -f {out_feat_norm_fp} -o {out_pix_dir} -m {1}  -ih {target_img_size} -iw {target_img_size} -rh {intermediate_img_size} -rw {intermediate_img_size} --force_update {force_update}')

print(len(video_names))
if args.use_mp:
    proc_queue = [{'idx': idx, 'vn': vn} for idx, vn in enumerate(video_names)]

    pool = Pool(processes=args.num_workers)
    tqdm_kwargs = dict(total=len(proc_queue), desc=f'Processing dataset into {args.out_data_dir}')

    def task_proxy(kwargs):
        return proc_video_by_name(**kwargs)

    for _ in tqdm(pool.imap_unordered(task_proxy, proc_queue), **tqdm_kwargs):
        pass
else:
    for idx, vn in enumerate(video_names):

        proc_video_by_name(idx, vn)


print('Done.')



# python process.py -i /cto_studio/vistring/liuyunfei/repo/dataset/HDTF -o /cto_studio/vistring/liuyunfei/repo/dataset/HDTF-semantic_mesh/
