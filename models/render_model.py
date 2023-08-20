import os
import os.path
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms.functional import crop
from torchvision.utils import save_image
from torch.nn.functional import interpolate
from torch.cuda.amp import autocast as autocast

from .componments.networks import init_net, get_scheduler
from .componments import render_G
from .componments.losses import GANLoss, MaskedL1Loss, VGGLoss
from .base_model import BaseModel


class RenderModel(BaseModel):
    def __init__(self, opt):
        """Initialize the Render Model class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        #torch.autograd.set_detect_anomaly(True)

        #super().__init__(opt)
        BaseModel.__init__(self, opt)

        self.Tensor = torch.cuda.FloatTensor
        self.n_frames_D = opt.n_frames_D if hasattr(opt, 'n_frames_D') is True else 1
        self.n_face_channels = 3
        self.n_mesh_channels = 3
        self.nrow = 4
        self.n_frame_channels = self.n_mesh_channels
        if opt.use_example > 0:
            self.n_frame_channels += self.n_face_channels * opt.use_example
        self.with_mouth_D = False #True if not self.opt.with_random_crop else False
        self.with_Mesh2Image = False
        self.dynamic_example = opt.dynamic_example
        if self.dynamic_example:
            self.dynamic_example_frame0_mode = opt.dynamic_example_frame0_mode
            self.dynamic_example_loss_mode = opt.dynamic_example_loss_mode
        self.use_position_encoding = opt.use_position_encoding
        self.num_encoding_functions = opt.num_encoding_functions

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        # define networks
        self.model_names = ['Render_G']
        self.ext_channel = 0
        if self.use_position_encoding:
            self.ext_channel += self.num_encoding_functions * 2
            if opt.include_input:
                print('===>With include_input<===')
                self.ext_channel += 1
            # self.ext_channel += self.num_encoding_functions * 2
        if self.dynamic_example and self.dynamic_example_frame0_mode==2:
            self.ext_channel += 3
        self.Render_G = init_net(render_G.Render_G(opt, ext_channel=self.ext_channel), init_type='normal', init_gain=0.02, gpu_ids=opt.gpu_ids)
        # define only during training time
        if self.isTrain:
            # define losses names
            self.loss_names_G = ['L1', 'VGG', 'Style', 'G_GAN', 'G_FM', ] #, "loss_FFL"]
            
            # criterion
            self.criterionMaskL1 = MaskedL1Loss().to(self.device)
            self.criterionL1 = nn.L1Loss().to(self.device)
            self.criterionVGG = VGGLoss().to(self.device)
            self.criterionFlow = nn.L1Loss().to(self.device)
            #self.ffl = FFL(loss_weight=1.0, alpha=1.0).to(self.device) # initialize nn.Module class

            # initialize optimizer G
            if opt.TTUR:
                beta1, beta2 = 0, 0.9
                lr = opt.lr / 2
            else:
                beta1, beta2 = opt.beta1, 0.999
                lr = opt.lr
            self.optimizer_G = torch.optim.Adam([{'params': self.Render_G.module.parameters(),
                                                  'initial_lr': lr}],
                                                lr=lr,
                                                betas=(beta1, beta2))
            self.optimizers.append(self.optimizer_G)

            # fp16 training
            if opt.fp16:
                self.scaler = torch.cuda.amp.GradScaler()

            # discriminator setting
            if not opt.no_discriminator:
                from .componments import render_D
                self.model_names += ['Render_D']
                self.Render_D = init_net(render_D.Render_D(
                    opt, ext_channel=self.ext_channel), init_type='normal', init_gain=0.02, gpu_ids=opt.gpu_ids)
                if self.with_mouth_D:
                    from .componments import render_mouth_D
                    self.model_names += ['Mouth_D']
                    self.Mouth_D = init_net(render_mouth_D.RenderMouth_D(opt), init_type='normal', init_gain=0.02, gpu_ids=opt.gpu_ids)

                self.criterionGAN = GANLoss(opt.gan_mode, tensor=self.Tensor).to(self.device)
                self.loss_names_D = ['D_real', 'D_fake']
                # initialize optimizer D
                if opt.TTUR:
                    beta1, beta2 = 0, 0.9
                    lr = opt.lr * 2
                else:
                    beta1, beta2 = opt.beta1, 0.999
                    lr = opt.lr
                self.optimizer_D = torch.optim.Adam([{'params': self.Render_D.module.netD.parameters(),
                                                      'initial_lr': lr}],
                                                    lr=lr,
                                                    betas=(beta1, beta2))
                self.optimizers.append(self.optimizer_D)

                if self.with_mouth_D:
                    self.optimizer_D_mouth = torch.optim.Adam([{'params': self.Mouth_D.module.netD.parameters(),
                                                      'initial_lr': lr}],
                                                    lr=lr,
                                                    betas=(beta1, beta2))
                    self.optimizers.append(self.optimizer_D_mouth)
        
        self.loss_names = self.loss_names_G + self.loss_names_D
        


    def init_paras(self, dataset):
        opt = self.opt
        iter_path = os.path.join(self.save_dir, 'iter.txt')
        start_epoch, epoch_iter = 1, 0
        ### if continue training, recover previous states
        if opt.continue_train:
            if os.path.exists(iter_path):
                start_epoch, epoch_iter = np.loadtxt(
                    iter_path, delimiter=',', dtype=int)
                print('Resuming from epoch %d at iteration %d' %
                      (start_epoch, epoch_iter))
                # change epoch count & update schedule settings
                opt.epoch_count = start_epoch
                self.schedulers = [get_scheduler(
                    optimizer, opt) for optimizer in self.optimizers]
                # print lerning rate
                lr = self.optimizers[0].param_groups[0]['lr']
                print('update learning rate: {} -> {}'.format(opt.lr, lr))
            else:
                print('not found training log, hence training from epoch 1')

        total_steps = (start_epoch-1) * len(dataset) + epoch_iter
        total_steps = total_steps // opt.print_freq * opt.print_freq

        return start_epoch, opt.print_freq, total_steps, epoch_iter

    def crop_mouth(self, image, mouth_bbox, mouth_size=256):
        nb, nc, _, _ = image.shape
        mouth_image = torch.zeros([nb, nc, mouth_size, mouth_size], device=image.device)
        for i in range(nb):
            bbox = mouth_bbox[i]
            mouth_crop = crop(image[i:i+1], bbox[0,1], bbox[0,0], bbox[1,1]-bbox[0,1], bbox[1,0]-bbox[0,0])
            mouth_image[i:i+1] = interpolate(mouth_crop, size=mouth_size)

        return mouth_image

    def set_input_pre(self, data, data_info=None):
        self.example_image_pre1 = data['example_pre1']
        if self.opt.use_example > 0:
            self.example_image_pre1 = torch.cat(self.example_image_pre1, dim=1).to(self.device)
        else:
            self.example_image_pre1 = torch.tensor([]).to(self.device)
        self.mesh_map_pre1, self.tgt_image_pre1 = data['mesh_pre1'], data['face_pre1']
        self.mesh_map_pre1 = self.mesh_map_pre1.to(self.device)
        self.tgt_image_pre1 = self.tgt_image_pre1.to(self.device)

    def set_offset_frame_names(self, offset_list, offset_face_names, offset_mesh_names):
        self.offset_list = offset_list
        self.offset_face_names = offset_face_names
        self.offset_mesh_names = offset_mesh_names
        self.pred_names = [ f"pred_{offset}" for offset in self.offset_list ]
        self.feature_map_names = [ f"feature_{offset}" for offset in self.offset_list ]

    def set_input(self, data, data_info=None):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        """
        
        self.example_image = data['example'][:, :3*self.opt.use_example, ...].to(self.device)

        self.mesh_map, self.tgt_image = data['mesh'], data['face']
        self.mesh_map = self.mesh_map.to(self.device)
        self.tgt_image = self.tgt_image.to(self.device)
        if self.use_position_encoding:
            self.position_encoding = data['time_position']
            self.position_encoding = self.position_encoding.to(self.device)
        
        if self.with_mouth_D:
            self.mouth_bbox = data['transformed_mouth_bbox'].type(torch.IntTensor)
            self.mouth_bbox = self.mouth_bbox.to(self.device)
            self.mouth_mesh = self.crop_mouth(self.mesh_map[:, -self.n_mesh_channels:, ...], self.mouth_bbox)
            self.mouth_image = self.crop_mouth(self.tgt_image[:, -self.n_face_channels:, ...], self.mouth_bbox)
            self.mouth_example = self.crop_mouth(self.example_image[:, -self.n_face_channels:, ...], self.mouth_bbox)


    def forward(self): #def process_input(self):
        ''' forward pass for feature2Face
        '''
        if self.use_position_encoding:
            self.input_feature_maps = torch.cat([self.mesh_map, self.example_image, self.position_encoding], dim=1)
        else:
            self.input_feature_maps = torch.cat([self.mesh_map, self.example_image], dim=1)

        self.fake_pred = self.Render_G(self.input_feature_maps)

        if self.with_mouth_D:
            self.mouth_feature_maps = torch.cat([self.mouth_mesh, self.mouth_example], dim=1)
            self.mouth_pred = self.crop_mouth(self.fake_pred, self.mouth_bbox)


    def backward_G(self):
        """Calculate GAN and other loss for the generator"""

        real_AB = torch.cat((self.input_feature_maps, self.tgt_image), dim=1)
        fake_AB = torch.cat((self.input_feature_maps, self.fake_pred), dim=1)

        pred_real = self.Render_D(real_AB)
        pred_fake = self.Render_D(fake_AB)
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        # L1, vgg, style loss
        loss_l1 = self.criterionL1(self.fake_pred, self.tgt_image) * self.opt.lambda_L1
        loss_vgg, loss_style = self.criterionVGG(self.fake_pred, self.tgt_image, style=True)
        loss_vgg = torch.mean(loss_vgg) * self.opt.lambda_feat
        loss_style = torch.mean(loss_style) * self.opt.lambda_feat
        loss_FM = self.compute_FeatureMatching_loss(pred_fake, pred_real) #* 10

        loss_list = [loss_l1, loss_vgg, loss_style, loss_G_GAN, loss_FM]
        self.loss_L1, self.loss_VGG, self.loss_Style, self.loss_G_GAN, self.loss_G_FM = loss_l1, loss_vgg, loss_style, loss_G_GAN, loss_FM

        # combine loss and calculate gradients
        if not self.opt.fp16:
            self.loss_G = loss_G_GAN + loss_l1 + loss_vgg + loss_style + loss_FM 
            self.loss_G.backward()
        else:
            with autocast():
                self.loss_G = loss_G_GAN + loss_l1 + loss_vgg + loss_style + loss_FM 
            self.scaler.scale(self.loss_G).backward()

        self.loss_dict = {**self.loss_dict, **dict(zip(self.loss_names_G, loss_list))}

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # GAN loss
        real_AB = torch.cat((self.input_feature_maps, self.tgt_image), dim=1)
        fake_AB = torch.cat((self.input_feature_maps, self.fake_pred), dim=1)
            
        pred_real = self.Render_D(real_AB)
        pred_fake = self.Render_D(fake_AB.detach())

        if self.with_mouth_D:
            mouth_real_AB = torch.cat([self.mouth_feature_maps, self.mouth_image], dim=1)
            mouth_fake_AB = torch.cat([self.mouth_feature_maps, self.mouth_pred], dim=1)
            mouth_pred_real = self.Mouth_D(mouth_real_AB)
            mouth_pred_fake = self.Mouth_D(mouth_fake_AB.detach())
        if self.opt.fp16:
            with autocast():
                loss_D_real = self.criterionGAN(pred_real, True)
                loss_D_fake = self.criterionGAN(pred_fake, False)
                if self.with_mouth_D:
                    loss_D_mouth_real = self.criterionGAN(mouth_pred_real, True)
                    loss_D_mouth_fake = self.criterionGAN(mouth_pred_fake, False)

        else:
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_fake = self.criterionGAN(pred_fake, False)
            if self.with_mouth_D:
                loss_D_mouth_real = self.criterionGAN(mouth_pred_real, True)
                loss_D_mouth_fake = self.criterionGAN(mouth_pred_fake, False)

        if self.with_mouth_D:
            lambda_mouth = 0.5
            loss_D_real = loss_D_real * (1-lambda_mouth) + loss_D_mouth_real * lambda_mouth
            loss_D_fake = loss_D_fake * (1-lambda_mouth) + loss_D_mouth_fake * lambda_mouth

        lambda_real = 0.666
        self.loss_D = loss_D_fake * (1-lambda_real) + loss_D_real * lambda_real

        self.loss_dict = dict(zip(self.loss_names_D, [loss_D_real, loss_D_fake]))

        self.loss_D_real, self.loss_D_fake = loss_D_real, loss_D_fake

        if not self.opt.fp16:
            self.loss_D.backward()
        else:
            self.scaler.scale(self.loss_D).backward() #retain_graph=True)

    def compute_FeatureMatching_loss(self, pred_fake, pred_real):
        # GAN feature matching loss
        loss_FM = torch.zeros(1).to(self.device)
        feat_weights = 4.0 / (self.opt.n_layers_D + 1)
        D_weights = 1.0 / self.opt.num_D
        for i in range(min(len(pred_fake), self.opt.num_D)):
            for j in range(len(pred_fake[i])):
                loss_FM += D_weights * feat_weights * \
                    self.criterionL1(
                        pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat

        return loss_FM

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
       
        self.forward()
        # update D
        # enable backprop for D
        self.set_requires_grad(self.Render_D, True)
        
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        
        if self.with_mouth_D:
            self.set_requires_grad(self.Mouth_D, True)
            self.optimizer_D_mouth.zero_grad()
        if not self.opt.fp16:
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights
            
            if self.with_mouth_D:
                self.optimizer_D_mouth.step()
        else:
            with autocast():
                self.backward_D()
            self.scaler.step(self.optimizer_D)
            
            if self.with_mouth_D:
                self.scaler.step(self.optimizer_D_mouth)

        # update G
        # D requires no gradients when optimizing G
        self.set_requires_grad(self.Render_D, False)
        if self.with_mouth_D:
            self.set_requires_grad(self.Mouth_D, False)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        if not self.opt.fp16:
            self.backward_G()                   # calculate graidents for G
            self.optimizer_G.step()             # udpate G's weights
        else:
            with autocast():
                self.backward_G()
            self.scaler.step(self.optimizer_G)
            self.scaler.update()


    def inference(self, mesh, example, flow=None):
        """ inference process """
        with torch.no_grad():
            if len(example) > 0:
                _example_image = torch.cat(example, dim=1).to(self.device)
            else:
                _example_image = torch.tensor([]).to(self.device)

            _mesh_map= mesh.to(self.device)
            input_feature_maps = torch.cat([_mesh_map, _example_image], dim=1)
            
            pred = self.Render_G(input_feature_maps)

        return pred
