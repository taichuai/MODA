import torch
import torch.nn as nn


from .networks import MultiscaleDiscriminator
from torch.cuda.amp import autocast as autocast


class Render_D(nn.Module):
    def __init__(self, opt, ext_channel=0):
        super().__init__()
        # initialize
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor
        self.tD = opt.n_frames_D #if opt.n_frames_D > 2 else 1 # n_frames_D=2 for flow
        self.output_nc = opt.output_nc
        n_face_channels = 3
        n_mesh_channels = 3
        n_flow_channel = 2
        n_example_channels = n_face_channels * opt.use_example
        input_nc = self.tD * (n_face_channels + n_mesh_channels) + n_example_channels + ext_channel
        if opt.n_frames_D == 2:
            input_nc += n_flow_channel

        # define networks
        self.netD = MultiscaleDiscriminator(
            input_nc, ndf=opt.ndf, n_layers=opt.n_layers_D, num_D=opt.num_D, getIntermFeat=not opt.no_ganFeat)

        print('----- Mesh2Face Discriminator networks initialized --------')
        print('-----------------------------------------------------------')

    #@autocast()
    def forward(self, input):
        if self.opt.fp16:
            with autocast():
                pred = self.netD(input)
        else:
            pred = self.netD(input)

        return pred
