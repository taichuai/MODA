import os
import cv2
import torch
import imageio
import argparse
import numpy as np
from tqdm import tqdm
from rich import print
from datetime import datetime
from skimage.io import imread
from models import create_model
from utils.util import tensor2im, im2tensor
from utils.util import assign_attributes, parse_config
from options.test_render_options import TestOptions as RenderOptions
from utils.util import merge_audio_to_video, prepare_video_data, get_positional_encoding


def reinit_opt(opt):
    if type(opt.dataset_names) == str:
        opt.dataset_names = [opt.dataset_names]
        opt.train_dataset_names = [x.split('.')[0] for x in sorted(os.listdir(os.path.join(opt.dataroot, 
                                    opt.dataset_name_train, 'render', 'pix2pix', 'feature'))) if x.endswith('.png')]
        opt.validate_dataset_names = [x.split('.')[0] for x in sorted(os.listdir(os.path.join(opt.dataroot, 
                                    opt.dataset_name_val, 'render', 'pix2pix', 'feature'))) if x.endswith('.png')]
    return opt


def main(opts):

    render_opt = RenderOptions().parse()  # get test options
    render_opt.batch_size = 1             # test code only supports batch_size = 1
    render_opt.phase = 'all'
    render_opt.serial_batches = True
    im_size = (512, 512)

    assign_attributes(opts, render_opt)
    render_opt = reinit_opt(render_opt)

    # create a model given opt.model and other options
    example_filenames = opts.example_fp
    if os.path.isfile(example_filenames):
        render_opt.example_filename = example_filenames
    model = create_model(render_opt)
    model.setup(render_opt)
    model.eval()

    vn_lst = [x for x in os.listdir(opts.video_dir) if '-feature' in x and '.mp4' in x]
    if len(vn_lst) == 0:
        vn_lst = [x for x in os.listdir(opts.video_dir) if '.mp4' in x]

    for vn in vn_lst:
        video_fp = os.path.join(opts.video_dir, vn)
        video_name = os.path.basename(video_fp).split('.')[0].replace('-feature', '')
        
        out_video_fp = os.path.join(opts.output_dir, video_name + f'{opts.sample_id:02d}' + '-render.mp4')
        if os.path.exists(out_video_fp): continue
        
        # Read frame from given video and condition frame
        temp_dir = os.path.join(os.path.dirname(video_fp), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        img_pwds, audio_fp = prepare_video_data(video_fp, temp_dir)
        os.makedirs(opts.output_dir,  exist_ok=True)

        tmp_video_fp = os.path.join(opts.output_dir, video_name + '-temp.mp4')
        writer = imageio.get_writer(tmp_video_fp, fps=30)
        
        if hasattr(opts, 'example_fp'):
            print(f'Loading example from {opts.example_fp}')
            if opts.example_fp.endswith('.pth'):
                im_example = torch.load(opts.example_fp)['examples'][0]
            else:
                im_example = cv2.imread(opts.example_fp)[..., ::-1]
                im_example = cv2.resize(im_example, (512, 512))
        else:
            print(f'Loading example from {opts.example_filename}')
            im_example = torch.load(render_opt.example_filename)['examples'][0]
        print('Reference shape:', im_example.shape)

        example = im2tensor(im_example).unsqueeze(0)
        
        for idx, mesh_fp in enumerate(tqdm(img_pwds, desc=f"test data {video_name}")):

            im_mesh = imread(mesh_fp)
            if im_mesh.shape[0] != im_size[0] or im_mesh.shape[1] != im_size[1]:
                im_mesh = cv2.resize(im_mesh, im_size)
            mesh = im2tensor(im_mesh).unsqueeze(0)
            
            if opts.use_position_encoding:
                t_embd = get_positional_encoding(idx, render_opt.position_encoding_size, example.shape[-1], render_opt.num_encoding_functions, render_opt.include_input)
                example_list = [example, t_embd]
            else:
                example_list = [example]
            
            pred_im_list = []
            pred_list = [model.inference(mesh, example_list)]
            pred_im_list = [ tensor2im(pred) for pred in pred_list ]

            concat_res = np.concatenate(pred_im_list + [im_example.copy(), im_mesh], axis=1)
            writer.append_data(concat_res)
        
        writer.close()
        
        merge_audio_to_video(audio_fp, tmp_video_fp, out_video_fp)

        # remove temporal files
        os.system(f'rm {tmp_video_fp}')
        os.system(f'rm -rf {temp_dir}')

        # print summary
        print(f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}: Fnished! Please check:\n\t", out_video_fp)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,  default='configs/renderer/renderer_wtpe.yaml', 
                        help='[!] Override the below parameters from config with cmd args')
    parser.add_argument('--force_up',       default=True)
    parser.add_argument('--sample_id',   type=int,  default=0)
    parser.add_argument('--video_dir',   default='test_results/HDTF_feature/Cathy')
    parser.add_argument('--output_dir',  type=str, default='test_results/Final/Cathy')
    args = parser.parse_args()

    opts = parse_config(args.config)
    assign_attributes(args, opts)

    main(opts)

    
