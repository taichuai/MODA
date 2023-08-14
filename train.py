import os
import copy
import torch
import argparse
from tqdm import tqdm
from rich import print
from models import create_model
from rich.progress import track
from datasets import create_dataset
from utils.visualizer import Visualizer
from utils.util import assign_attributes, parse_config
from options.train_feature_options import TrainOptions


parser = argparse.ArgumentParser()
# parser.add_argument('--config',  type=str,  default='config/train_audio2feature_dualv2_LSP.yaml')
# parser.add_argument('--config',  type=str,  default='config/finetune_meshrefiner_zhubo.yaml')
# parser.add_argument('--config',  type=str,  default='config/train_mp2flameparam.yaml')
parser.add_argument('--config',  type=str,  default='config/train_meshrefiner_flame.yaml')
args = parser.parse_args()

train_opt = TrainOptions().parse()

opts = parse_config(args.config)
assign_attributes(opts, train_opt)
train_opt.teacher = parse_config(train_opt.teacher) if os.path.exists(train_opt.teacher) else None

if opts.dataset_mode in ('audiovisual_v5', 'mediapipe2flame', 'flame_composite'): train_opt.subject_name = opts.train_sub_names
dataset_train = create_dataset(train_opt)
train_opt.n_subjects = dataset_train.dataset.n_subjects
if not train_opt.motion_mode:
    train_opt.vertice_dim = dataset_train.dataset.vertices_len * 3
if 'once' in train_opt.model:
    train_opt.lip_vertice_dim = dataset_train.dataset.lip_vertices_len * 3
    train_opt.eye_vertice_dim = dataset_train.dataset.eye_vertices_len * 3
    print(train_opt.eye_vertice_dim)
val_opt = copy.deepcopy(train_opt)
val_opt.data_mode = 'Val'
if opts.dataset_mode in ('audiovisual_v5', 'mediapipe2flame', 'flame_composite'): val_opt.subject_name = opts.test_sub_names
dataset_val = create_dataset(val_opt)

len_train = len(dataset_train)
len_val = len(dataset_val)

print('### Training   dataset len:', len_train)
print('### Validation dataset len:', len_val)
print('### N subjects:', train_opt.n_subjects)

model = create_model(train_opt)
model.setup(train_opt)
model.print_networks(verbose=True)
if os.path.exists(train_opt.load_epoch):
    model.load_networks(train_opt.load_epoch)

viz = Visualizer(train_opt)

min_val_error = 1e8
for epoch in range(train_opt.n_epochs):
    model.train()
    step = 0
    t_bar = tqdm(dataset_train, desc=f'Training   (epoch={epoch:02d}/{train_opt.n_epochs})')
    for data in t_bar:
        model.set_input(data)
        model.optimize_parameters()

        losses = model.get_current_losses()
        viz.plot_current_errors({'train/'+ k: v for k, v in losses.items()})
        if step % 500 == 0:
            # print(losses)
            show_losses = ''
            for k, v in losses.items():
                show_losses += f' {k}: {v:.4f}'
            t_bar.set_description(f'Training   (epoch={epoch:02d}/{train_opt.n_epochs}) | {show_losses} ==>')
    
    cur_lr = model.update_learning_rate()
    viz.logger.log({'lr': cur_lr})
    
    model.eval()
    cur_val_errors = []
    for data in track(dataset_val,   description=f'Validation (epoch={epoch:02d}/{train_opt.n_epochs})'):
        with torch.no_grad():
            model.set_input(data)
            model.validate()

            losses = model.get_current_losses()
        viz.plot_current_errors({'val/'+ k: v for k, v in losses.items()})
        cur_val_errors.append(losses['total'])
        step += 1
    
    if epoch > 40:
        cur_val_error_ave = sum(cur_val_errors) / len(cur_val_errors)
        if cur_val_error_ave < min_val_error:
            min_val_error = cur_val_error_ave
            model.save_networks(epoch=epoch, is_best=True)
            print(f'Best model saved. epoch={epoch}, val_error={min_val_error}')

        if (epoch + 1) % 10 == 0 or 'finetune' in train_opt.model.lower():
            model.save_networks(epoch=epoch)
            print(f'Model saved. epoch={epoch}')
    



# Once: python train.py --task Audio2FeatureOnce --batch_size 32 --dataset_mode audiovisual_v3 --model audio2featureonce
# OnceV1: python train.py --task Audio2FeatureOnce --batch_size 32 --dataset_mode audiovisual_v4 --model audio2featureoncev1 
# Headmotion: python train.py --batch_size 32 --motion_mode 1 --task Audio2FeatureLSTM_HeadmotionCD --classifier_decoder 1/0
# Vertices: python train.py --task Audio2FeatureLSTM --batch_size 32
# MeshRefiner: python train.py --task MeshRefiner --batch_size 32 --dataset_mode audiovisual_v3 --model meshrefiner
# Once with former: python train.py --task Audio2FeatureOnceWithTorsoFormerMeshID --batch_size 4 --dataset_mode audiovisual_v4 --model audio2featureoncev1 --body base --subject_head mesh --feature_decoder Former --max_seq_len 360 --time_frame_length 360 --n_epochs 500
# FormerOnce: python train.py --task Audio2FeatureOnceWithTorsoFormerOnceMeshID --batch_size 32 --dataset_mode audiovisual_v4 --model audio2featureoncev1 --body base --subject_head mesh --feature_decoder FormerOnce --n_epochs 6000

# python train.py --task Audio2FeatureOnceWithTorsoFormerDualV2MeshID --batch_size 48 --num_threads 16 --dataset_mode audiovisual_v4 --model audio2featureoncev1 --body dualv2 --subject_head mesh --feature_decoder FormerOnce --n_epochs 16000