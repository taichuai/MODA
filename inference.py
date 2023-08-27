import os
import yaml
import argparse
from easydict import EasyDict


def gen_feature(audio_fp_or_dir, cfg, out_dir, n_sample):
    if os.path.isdir(audio_fp_or_dir):
        audio_files = [os.path.join(audio_fp_or_dir, x) for x in os.listdir(audio_fp_or_dir)]
    else:
        audio_files = [audio_fp_or_dir]

    audio_files_str = ','.join(audio_files)

    out_dir_lst = []

    for i in range(n_sample):
        t_out_dir = os.path.join(out_dir, f'{i:02d}')

        print('#'*60)
        print('#'*5, f'Processing [{i}/{n_sample}]')
        print('#'*60)

        cmd_str = f'python scripts/generate_feature.py --config {cfg.feat.cfg_fp} ' + \
                f'--meta_dir {cfg.feat.meta_dir} --test_person {cfg.person_name} ' + \
                f'--driven_audios {audio_files_str} --out_dir {t_out_dir} ' + \
                f'--a2f_ckpt_fp {cfg.feat.a2f_ckpt_fp} ' + \
                f'--refiner_ckpt_fp {cfg.feat.refiner_ckpt_fp} '
        
        os.system(cmd_str)
        out_dir_lst.append(os.path.join(t_out_dir, cfg.person_name))
    
    return out_dir_lst


def render_video(input_dirs, output_dir, cfg):
    for (idx, input_dir) in enumerate(input_dirs):
        print('#'*60)
        print('#'*5, f'Processing [{idx}/{len(input_dirs)} => {input_dir}]')
        print('#'*60)

        cmd_str = f'python scripts/generate_video.py --config {cfg.renderer.cfg_fp} ' + \
                f'--video_dir {input_dir} --output_dir {output_dir} ' + \
                f'--sample_id {idx}'
        
        os.system(cmd_str)


parser = argparse.ArgumentParser('Inference entrance for MODA.')
parser.add_argument('--audio_fp_or_dir', type=str, default='assets/data/test_audios')
parser.add_argument('--person_config',   type=str, default='configs/Cathy.yaml')
parser.add_argument('--output_dir',      type=str, default='results')
parser.add_argument('--n_sample',        type=int, default=2)

args = parser.parse_args()

os.system('export PYTHONPATH=.')

with open(args.person_config, 'r') as fid:
    person_cfg = EasyDict(yaml.safe_load(fid))

os.system('export PYTHONPATH=.')

inter_dirs = gen_feature(args.audio_fp_or_dir, person_cfg,
                         os.path.join(args.output_dir, 'inter'), args.n_sample)

render_video(inter_dirs, args.output_dir, person_cfg)

print(f'All done. Please check {args.output_dir} .')





