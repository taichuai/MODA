import torch.nn as nn

from .networks import RenderGenerator_Unet, RenderGenerator_normal, RenderGenerator_large

from torch.cuda.amp import autocast as autocast


class Render_G(nn.Module):
    def __init__(self, opt, ext_channel=0):
        super().__init__()
        # initialize
        self.opt = opt
        self.isTrain = opt.isTrain
        #self.tD = 2 if hasattr(opt, 'n_frames_D') and opt.n_frames_D==2 or hasattr(opt, 'with_flow') and opt.with_flow == True else 1 # for flow
        self.tD = 1
        n_mesh_channels = 3
        n_face_channels = 3
        n_flow_channels = 2
        n_example_channels = n_face_channels * opt.use_example
        input_nc = n_mesh_channels + n_example_channels + ext_channel
        #input_nc = self.tD * n_mesh_channels + n_example_channels
        if self.tD == 2:
            input_nc += n_flow_channels

        # define net G
        if opt.size == 'small':
            self.netG = RenderGenerator_Unet(
                input_nc=input_nc, output_nc=n_face_channels, num_downs=opt.n_downsample_G, ngf=opt.ngf)
        elif opt.size == 'normal':
            self.netG = RenderGenerator_normal(
                input_nc=input_nc, output_nc=n_face_channels, num_downs=opt.n_downsample_G, ngf=opt.ngf)  # test: just use 1 channel face parsing mask as input
            # input_nc=13, output_nc=3, num_downs=opt.n_downsample_G, ngf=opt.ngf)
        elif opt.size == 'large':
            self.netG = RenderGenerator_large(
                input_nc=input_nc, output_nc=n_face_channels, num_downs=opt.n_downsample_G, ngf=opt.ngf)

        print('---------- Generator networks initialized -------------')
        print('-------------------------------------------------------')

    def forward(self, input):
        if self.opt.fp16:
            with autocast():
                fake_pred = self.netG(input)
        else:
            fake_pred = self.netG(input)

        return fake_pred
