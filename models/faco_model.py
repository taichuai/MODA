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
        """Initialize the FaCo Model class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.Tensor = torch.cuda.FloatTensor
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        # define networks 
        self.model_names = ['FaCo_G']
        self.FaCo_G = networks.init_net(faco_net.FaCoNet(opt), init_type='normal', init_gain=0.02, gpu_ids=opt.gpu_ids)

        # define only during training time
        if self.isTrain:
            # define losses names
            self.loss_names = ['L1', 'GAN', 'total']    
            self.loss_names_G = ['L1', 'GAN', 'total']    
            # criterion
            self.criterionL1 = nn.L1Loss().cuda()
            
            # initialize optimizer G 
            if opt.TTUR:                
                beta1, beta2 = 0, 0.9
                lr = opt.lr / 2
            else:
                beta1, beta2 = opt.beta1, 0.999
                lr = opt.lr   

            self.optimizer_G = torch.optim.Adam([{'params': self.FaCo_G.parameters(),
                                                  'initial_lr': lr}], lr=lr, betas=(beta1, beta2))
            self.optimizers.append(self.optimizer_G)
            
            # discriminator setting
            if not opt.no_discriminator:
                self.model_names += ['D']
                from .componments.faco_net import PointDiscriminator
                self.D = networks.init_net(PointDiscriminator(opt), init_type='normal', init_gain=0.02, gpu_ids=opt.gpu_ids)

                from .componments.losses import GANLoss
                self.criterionGAN = GANLoss(opt.gan_mode, tensor=self.Tensor) 
                self.loss_names += ['D_real', 'D_fake']
                self.loss_names_D = ['D_real', 'D_fake']    
                # initialize optimizer D
                if opt.TTUR:                
                    beta1, beta2 = 0, 0.9
                    lr = opt.lr * 2
                else:
                    beta1, beta2 = opt.beta1, 0.999
                    lr = opt.lr
                self.optimizer_D = torch.optim.Adam([{'params': self.D.parameters(),
                                                      'initial_lr': lr}], lr=lr, betas=(beta1, beta2))
                self.optimizers.append(self.optimizer_D)

    
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
            self.subject_id = one_hot.to(self.device)
        else:
            self.subject_id = meta['MeanPoint'].reshape(meta['MeanPoint'].shape[0], -1).to(self.device)
        
        b, seq, t_dim = vertices_info.shape[:3]
        vertices_info = vertices_info.reshape((b, seq, t_dim//3, 3))

        self.input_lip = vertices_info[:, :, meta['Lip']].reshape((b, seq, -1))
        self.input_eye = vertices_info[:, :, meta['Eye']].reshape((b, seq, -1))

        self.target = vertices_info.reshape((b, seq, -1)).to(self.device)

    def forward(self):
        ''' forward pass for feature2Face
        '''  
        self.fake_pred = self.FaCo_G(self.input_lip, self.input_eye, self.subject_id)

    def backward_G(self):
        """Calculate GAN and other loss for the generator"""
        # GAN loss
        
        self.calculate_loss()
        self.loss_G.backward()
        
        self.loss_dict = {**self.loss_dict, **dict(zip(self.loss_names_G, [self.loss_L1, self.loss_GAN]))}

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # GAN loss
        real_AB = torch.cat((self.input, self.target),    dim=-1)
        fake_AB = torch.cat((self.input, self.fake_pred), dim=-1)
        pred_real = self.D(real_AB, self.subject_id)
        pred_fake = self.D(fake_AB.detach(), self.subject_id)
        
        with autocast():
            loss_D_real = self.criterionGAN(pred_real, True) * 2
            loss_D_fake = self.criterionGAN(pred_fake, False)
        
        self.loss_D = (loss_D_fake + loss_D_real) * 0.5 
        
        self.loss_D_real = loss_D_real
        self.loss_D_fake = loss_D_fake
        
        self.loss_dict = dict(zip(self.loss_names_D, [loss_D_real, loss_D_fake]))   
        
        self.loss_D.backward()
    
    def calculate_loss(self):
        real_AB = torch.cat((self.input, self.target),    dim=-1)
        fake_AB = torch.cat((self.input, self.fake_pred), dim=-1)
        pred_real = self.MeshRefiner_D(real_AB, self.subject_id)
        pred_fake = self.MeshRefiner_D(fake_AB, self.subject_id)
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        # L1, vgg, style loss
        loss_l1 = self.criterionL1(self.fake_pred, self.target) * self.opt.lambda_L1
        
        self.loss_G = loss_G_GAN + loss_l1

        self.loss_L1 = loss_l1
        self.loss_GAN = loss_G_GAN
        self.loss_total = self.loss_G
    
    def compute_FeatureMatching_loss(self, pred_fake, pred_real):
        # GAN feature matching loss
        loss_FM = torch.zeros(1).cuda()
        feat_weights = 4.0 / (self.opt.n_layers_D + 1)
        D_weights = 1.0 / self.opt.num_D
        for i in range(min(len(pred_fake), self.opt.num_D)):
            for j in range(len(pred_fake[i])):
                loss_FM += D_weights * feat_weights * \
                    self.criterionL1(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
        
        return loss_FM
    
    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        # only train single image generation
        ## forward
        self.forward()
        # update D
        self.set_requires_grad(self.MeshRefiner_D, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero    
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
            
        # update G
        self.set_requires_grad(self.MeshRefiner_D, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    @torch.no_grad()
    def inference(self, x_in_lip, x_in_eye, n_subjects, sub_id):
        """ inference process """
        b, seq, t_dim = x_in_lip.shape[:3]

        x_in_lip = x_in_lip.reshape((b, seq, -1)).to(self.device)
        x_in_eye = x_in_eye.reshape((b, seq, -1)).to(self.device)
        
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
            fake_pred = self.FaCo_G(x_in_lip, x_in_eye, sub_info) 
        return fake_pred.detach().squeeze().cpu().numpy()

    @torch.no_grad()
    def validate(self):
        """ validate process """
        self.forward()
        self.calculate_loss()








