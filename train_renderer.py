import os
import copy
import torch
import argparse
import numpy as np
from tqdm import tqdm
from rich import print
from models import create_model
from rich.progress import track
from datasets import create_dataset
from utils.visualizer import Visualizer
from options.train_render_options import TrainOptions
from utils.util import assign_attributes, parse_config


def reinit_opt(opt):
    if type(opt.dataset_names) == str:
        try:
            opt.dataset_names = [opt.dataset_names]
            opt.train_dataset_names = np.loadtxt(os.path.join(opt.dataroot, 
                                                            opt.dataset_names[0], 'render',
                                                            opt.train_dataset_names), dtype=np.str).tolist()
            opt.validate_dataset_names = np.loadtxt(os.path.join(opt.dataroot, 
                                                                opt.dataset_names[0], 'render',
                                                                opt.validate_dataset_names), dtype=np.str).tolist()
            if type(opt.validate_dataset_names) == str:
                opt.validate_dataset_names = [opt.validate_dataset_names]
        except Exception as e:
            print(e)
    
    if type(opt.cand_dataset_names) == str:
        opt.cand_dataset_names = [opt.cand_dataset_names]
    return opt


parser = argparse.ArgumentParser()
parser.add_argument('--config',  type=str,  default='configs/train/renderer/Cathy.yaml')
args = parser.parse_args()

train_opt = TrainOptions().parse()

opts = parse_config(args.config)
assign_attributes(opts, train_opt)
train_opt = reinit_opt(train_opt)

dataset_train = create_dataset(train_opt)

val_opt = copy.deepcopy(train_opt)
val_opt.data_mode = val_opt.dataset_type = 'Val'
dataset_val = create_dataset(val_opt)

len_train = len(dataset_train)
len_val = len(dataset_val)

print('### Training   dataset len:', len_train)
print('### Validation dataset len:', len_val)

model = create_model(train_opt)
model.setup(train_opt)
model.print_networks(verbose=True)

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
            if len(show_losses) > 70: show_losses = show_losses[:70] + ' ...'
            t_bar.set_description(f'Training   (epoch={epoch:02d}/{train_opt.n_epochs}) | {show_losses} ==>')
        
    
    cur_lr = model.update_learning_rate()
    viz.logger.log({'lr': cur_lr})

    visuals = model.get_current_visuals()
    viz.save_images(os.path.join(viz.log_dir, 'image'), visuals, image_path=f'epoch-{epoch}')
    
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
    
    cur_val_error_ave = sum(cur_val_errors) / len(cur_val_errors)
    if cur_val_error_ave < min_val_error:
        min_val_error = cur_val_error_ave
        model.save_networks(epoch=epoch, is_best=True)
        print(f'Best model saved. epoch={epoch}, val_error={min_val_error}')

    if (epoch + 1) % 10 == 0 or 'finetune' in train_opt.model.lower():
        model.save_networks(epoch=epoch)
        print(f'Model saved. epoch={epoch}')
    
