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
from .base_model import BaseModel


class RenderModel(BaseModel):
    def __init__(self, opt):
        """Initialize the Render class.

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
        
        self.with_optical_flow = True if self.n_frames_D == 2 else False
        if hasattr(opt, 'with_flow') and opt.with_flow == True:
            self.with_optical_flow = True
            self.n_frames_D = 2

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
        # self.feature_map, self.cand_image, self.tgt_image, self.facial_mask = \
        #     data['feature_map'], data['cand_image'], data['tgt_image'], data['weight_mask']
        self.example_image = data['example']
        if self.opt.use_example > 0:
            self.example_image = torch.cat(self.example_image, dim=1).to(self.device)
        else:
            self.example_image = torch.tensor([]).to(self.device)

        self.mesh_map, self.tgt_image = data['mesh'], data['face']
        self.mesh_map = self.mesh_map.to(self.device)
        self.tgt_image = self.tgt_image.to(self.device)
        if self.use_position_encoding:
            self.position_encoding = data['time_position']
            self.position_encoding = self.position_encoding.to(self.device)
        if self.n_frames_D > 1:
            for face_name in self.offset_face_names:
                if face_name == 'face_0': continue
                setattr(self, face_name, data[face_name].to(self.device))
            for mesh_name in self.offset_mesh_names:
                if mesh_name == 'mesh_0': continue
                setattr(self, mesh_name, data[mesh_name].to(self.device))

        if self.with_Mesh2Image:
            self.param = data['param'].float()
            self.param = self.param.to(self.device)
            #self.input_mesh = data['mean_dense_vertices']
            #self.input_mesh = self.input_mesh.to(self.device)
        if self.with_mouth_D:
            self.mouth_bbox = data['transformed_mouth_bbox'].type(torch.IntTensor)
            self.mouth_bbox = self.mouth_bbox.to(self.device)
            self.mouth_mesh = self.crop_mouth(self.mesh_map[:, -self.n_mesh_channels:, ...], self.mouth_bbox)
            self.mouth_image = self.crop_mouth(self.tgt_image[:, -self.n_face_channels:, ...], self.mouth_bbox)
            self.mouth_example = self.crop_mouth(self.example_image[:, -self.n_face_channels:, ...], self.mouth_bbox)

        if self.dynamic_example:
            self.set_input_pre(data)

    def forward_pre(self):
        if self.dynamic_example and self.dynamic_example_frame0_mode==0:
            self.example_image_pre1 = torch.zeros_like(self.example_image_pre1) - 1     #[3, 3, 512, 512]
            # self.mouth_example_pre1 = torch.zeros_like(self.mouth_example_pre1)
        elif self.dynamic_example and self.dynamic_example_frame0_mode==2:
            self.example_image_pre1 = torch.cat([torch.zeros_like(self.example_image_pre1) - 1, self.example_image_pre1], dim=1)    #[3, 6, 512, 512]

        self.input_feature_maps_pre1 = torch.cat([self.mesh_map_pre1, self.example_image_pre1], dim=1)

        self.fake_pred_pre1 = self.Render_G(self.input_feature_maps_pre1)

    def forward(self): #def process_input(self):
        ''' forward pass for feature2Face
        '''
         # t0 frame inference:
        if self.dynamic_example:
            self.forward_pre()
            # self.example_image = self.fake_pred_pre1.detach()
            
            if self.dynamic_example and self.dynamic_example_frame0_mode==2:
                self.example_image = torch.cat([self.fake_pred_pre1.data, self.example_image], dim=1)
            else:
                self.example_image = self.fake_pred_pre1.data

        if self.with_Mesh2Image:
            pred_mesh_feature = self.Mesh2Image(self.param)
            pred_mesh_feature = interpolate(pred_mesh_feature, size=512)
            self.input_feature_maps = torch.cat([pred_mesh_feature, self.example_image], dim=1)
        elif self.use_position_encoding:
            self.input_feature_maps = torch.cat([self.mesh_map, self.example_image, self.position_encoding], dim=1)
        else:
            self.input_feature_maps = torch.cat([self.mesh_map, self.example_image], dim=1)
            '''for i in range(self.n_frames_D):
                if self.offset_mesh_names[i] == 'mesh_0': continue
                mesh = getattr(self, self.offset_mesh_names[i])
                feature_map = torch.cat([mesh, self.example_image], dim=1)
                setattr( self, self.pred_names[i], feature_map )
                #self.input_feature_maps = torch.cat([feature_map, self.input_feature_maps], dim=1)'''

        if self.with_optical_flow:
            prev_face = getattr(self, self.offset_face_names[-2])
            self.tgt_flow = self.raft(self.tgt_image, prev_face)[-1]
            self.fake_pred = self.Render_G( self.input_feature_maps ) #torch.cat([self.tgt_flow.detach(), self.input_feature_maps], dim=1) )
            self.pred_flow = self.raft(self.fake_pred, prev_face)[-1]
        else:
            self.fake_pred = self.Render_G(self.input_feature_maps)

        if self.with_mouth_D:
            self.mouth_feature_maps = torch.cat([self.mouth_mesh, self.mouth_example], dim=1)
            self.mouth_pred = self.crop_mouth(self.fake_pred, self.mouth_bbox)

        '''if not self.with_optical_flow and self.n_frames_D > 1:
            input_feature_maps = torch.cat([ getattr(self, mesh_name) for mesh_name in self.offset_mesh_names[:-1] ] + [input_feature_maps], dim=1)
            multi_tgt_image = torch.cat([ getattr(self, face_name) for face_name in self.offset_face_names[:-1] ] + [self.tgt_image], dim=1)
            real_AB = torch.cat((input_feature_maps, multi_tgt_image), dim=1)
        else:
            real_AB = torch.cat((input_feature_maps, self.tgt_image), dim=1)

        return input_feature_maps, real_AB'''


    def inference(self, mesh, example, flow=None):
        """ inference process """
        with torch.no_grad():
            if len(example) > 0:
                _example_image = torch.cat(example, dim=1).to(self.device)
            else:
                _example_image = torch.tensor([]).to(self.device)

            if self.with_Mesh2Image:
                _mesh_param = mesh.to(self.device)
                pred_mesh_feature = self.Mesh2Image(_mesh_param)
                pred_mesh_feature = interpolate(pred_mesh_feature, size=512)
                input_feature_maps = torch.cat([pred_mesh_feature, self.example_image], dim=1)
            else:
                _mesh_map= mesh.to(self.device)
                input_feature_maps = torch.cat([_mesh_map, _example_image], dim=1)
            if self.with_optical_flow:
                assert flow is not None
                flow = flow.to(self.device)
                input_feature_maps = torch.cat([flow, input_feature_maps], dim=1)
            pred = self.Render_G(input_feature_maps)

        return pred
