import os
import torch
import numpy as np
import torch.nn as nn

from .componments import networks
from .base_model import BaseModel
from .componments import faco_net
from torch.cuda.amp import autocast as autocast


class FaCoModel(BaseModel):          
    def __init__(self, opt):
        """Initialize the Feature2Face class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.Tensor = torch.cuda.FloatTensor
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        # define networks 
        self.model_names = ['FaCo_G']
        self.FaCo_G = networks.init_net(faco_net.FaCoNet(opt), init_type='normal', init_gain=0.02, gpu_ids=opt.gpu_ids)

    
    def init_paras(self, dataset):
        opt = self.opt
        iter_path = os.path.join(self.save_dir, 'iter.txt')
        start_epoch, epoch_iter = 1, 0
        ### if continue training, recover previous states
        if opt.continue_train:        
            if os.path.exists(iter_path):
                start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)        
                print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))   
                # change epoch count & update schedule settings
                opt.epoch_count = start_epoch
                self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
                # print lerning rate
                lr = self.optimizers[0].param_groups[0]['lr']
                print('update learning rate: {} -> {}'.format(opt.lr, lr))
            else:
                print('not found training log, hence training from epoch 1')
                

        total_steps = (start_epoch-1) * len(dataset) + epoch_iter
        total_steps = total_steps // opt.print_freq * opt.print_freq  
        
        return start_epoch, opt.print_freq, total_steps, epoch_iter
    
    
    def set_input(self, data, data_info=None):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        """
        _, one_hot, self.av_rate, vertices_info, _, _, _, meta  = data  #[:4]
        
        for k, v in meta.items():
            if k in ['MeanPoint']: continue
            meta[k] = [x[0] for x in v]
        if self.opt.subject_head == 'onehot':
            self.subject_id  = one_hot.to(self.device)
        else:
            self.subject_id  = meta['MeanPoint'].reshape(meta['MeanPoint'].shape[0], -1).to(self.device)
        
        b, seq, t_dim = vertices_info.shape[:3]
        vertices_info = vertices_info.reshape((b, seq, t_dim//3, 3))

        self.input = torch.cat([
            vertices_info[:, :, meta['Lip']].reshape((b, seq, -1)), 
            vertices_info[:, :, meta['Eye']].reshape((b, seq, -1))], dim=-1).to(self.device)

        self.target = vertices_info.reshape((b, seq, -1)).to(self.device)


    def forward(self):
        ''' forward pass for feature2Face
        '''  
        self.fake_pred = self.PointRefiner_G(self.input, self.subject_id)

    def inference(self, x_in_lip, x_in_eye, n_subjects, sub_id):
        """ inference process """
        b, seq, t_dim = x_in_lip.shape[:3]

        x_in = torch.cat([
            x_in_lip.reshape((b, seq, -1)), 
            x_in_eye.reshape((b, seq, -1))], dim=-1).to(self.device)
        
        if self.opt.subject_head == 'onehot':
            one_hot_labels = np.eye(n_subjects)
            one_hot = one_hot_labels[sub_id]
            one_hot = np.reshape(one_hot,(-1,one_hot.shape[0]))
            one_hot = torch.FloatTensor(one_hot).to(device=self.device)
            sub_info = one_hot
        elif self.opt.subject_head == 'point':
            t_sub    = torch.from_numpy(sub_id)
            sub_info = t_sub.unsqueeze(0).float()
            sub_info = sub_info.reshape(sub_info.shape[0], -1).to(self.device)
        
        with torch.no_grad():      
            fake_pred = self.FaCo_G(x_in, sub_info) 
        return fake_pred.detach().squeeze().cpu().numpy()

    @torch.no_grad()
    def validate(self):
        """ validate process """
        self.forward()
        self.calculate_loss()








