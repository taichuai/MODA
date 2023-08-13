"""
Step-0: Extract audio file from video

===

Extract audio to wav file from the video file.
If set crop flag, crop the video and process it to 30fps
Depends:
    rich
    ffmpeg
    opencv-python
"""

__author__ = 'dreamtale'

import os
import cv2
import imageio
import argparse
from rich import print
from rich.progress import track

parser = argparse.ArgumentParser(description='Extract audio info')
parser.add_argument('-i', '--in_video_fp', type=str, help='Input video file path.')
parser.add_argument('-a', '--out_audio_fp', type=str, help='Output audio file path.')
parser.add_argument('-v', '--out_video_fp', type=str, help='Output video file path.')
parser.add_argument('--crop', action='store_true', help='Crop the video')
parser.add_argument('--target_fps', type=int, default=30,  help='target video fps')
parser.add_argument('--target_h',   type=int, default=512, help='target frame height')
parser.add_argument('--target_w',   type=int, default=512, help='target frame width')

args = parser.parse_args()

if os.path.isdir(args.out_audio_fp):
    args.out_audio_fp = os.path.join(args.out_audio_fp, os.path.basename(args.in_video_fp).spilt('.')[0] + '-audio.wav')

try:
    os.makedirs(os.path.dirname(args.out_audio_fp), exist_ok=True)
except Exception as e:
    print(e)

tmp_dir = '/home/liuyunfei/repo/lsp_dataset_preparation/tmp'
os.makedirs(tmp_dir, exist_ok=True)

vfp = os.path.join(tmp_dir, os.path.basename(args.in_video_fp).split('.')[0] + f'-{args.target_fps}fps.mp4')
cvt_wav_cmd = 'ffmpeg -i ' + args.in_video_fp + f' -filter:v fps=fps={args.target_fps} ' + vfp + ' -y'
print(f'<RUN> => {cvt_wav_cmd}')
os.system(cvt_wav_cmd)

cvt_wav_cmd = 'ffmpeg -i ' + vfp + f' -vf scale={args.target_h}:{args.target_w} -crf 2 ' + args.out_video_fp + ' -y'
print(f'<RUN> => {cvt_wav_cmd}')
os.system(cvt_wav_cmd)

extract_wav_cmd = 'ffmpeg -i ' + vfp + ' -f wav -ar 16000 ' + args.out_audio_fp + ' -y'
print(f'<RUN> => {extract_wav_cmd}')
os.system(extract_wav_cmd)

if args.crop:
    # only support CN
    cap = cv2.VideoCapture(args.in_video_fp)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps_ori = cap.get(cv2.CAP_PROP_FPS)
    fps_tar = args.target_fps
    sample_rate = fps_tar / fps_ori

    annotation_fp = args.in_video_fp.split('.')[0].split('-clip-')[0].replace('test-clips', 'annotation') + '.txt'
    with open(annotation_fp) as fid:
        line = fid.readlines()[0].strip()
        bbox = [float(x) for x in line.split(' ')]
    
    writer = imageio.get_writer(args.out_video_fp, fps=args.target_fps)
    t_idx = 0
    for i in track(range(int(n_frames))):
        ret, frame = cap.read()
        if not ret: break
        tar_idx = i * sample_rate
        l = int(bbox[0] * frame.shape[1])
        t = int(bbox[1] * frame.shape[0])
        r = int(bbox[2] * frame.shape[1])
        b = int(bbox[3] * frame.shape[0])
        frame_roi = frame[t:b, l:r, ...]
        frame_roi_rsz = cv2.resize(frame_roi, (args.target_w, args.target_h))
        while t_idx < tar_idx:
            writer.append_data(frame_roi_rsz[..., ::-1])
            t_idx += 1

    writer.close()
    print(f'Done. video is saved at {args.out_video_fp} .')
