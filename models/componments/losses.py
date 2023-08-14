import cv2
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor        
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None        
        gpu_id = input.get_device()
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).cuda(gpu_id).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).cuda(gpu_id).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)                
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class VGGLoss(nn.Module):
    def __init__(self, model=None):
        super(VGGLoss, self).__init__()
        if model is None:
            self.vgg = Vgg19()
        else:
            self.vgg = model

        self.vgg.cuda()
        # self.vgg.eval()
        self.criterion = nn.L1Loss()
        self.style_criterion = StyleLoss()
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.style_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        # self.weights = [5.0, 1.0, 0.5, 0.4, 0.8]
        # self.style_weights = [10e4, 1000, 50, 15, 50]

    def forward(self, x, y, style=False):
        x_vgg, y_vgg = self.vgg(x[:, :3, ...]), self.vgg(y[:, :3, ...])
        loss = 0
        if style:
            # return both perceptual loss and style loss.
            style_loss = 0
            for i in range(len(x_vgg)):
                this_loss = (self.weights[i] *
                             self.criterion(x_vgg[i], y_vgg[i].detach()))
                this_style_loss = (self.style_weights[i] *
                                   self.style_criterion(x_vgg[i], y_vgg[i].detach()))
                loss += this_loss
                style_loss += this_style_loss
            return loss, style_loss

        for i in range(len(x_vgg)):
            this_loss = (self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach()))
            loss += this_loss
        return loss
    

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x, y, m):
        t = (x - y)**2
        t = t.reshape(t.shape[0], -1) / t.shape[-1]
        t = torch.mean(t, 1)
        div = torch.sum(m)
        if div < 0.5: div = t.shape[0]
        ret = torch.sum(t * m.squeeze().float()) / div.float()
        return ret


class MotionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, seq_1, seq_2):
        """
        seq_1: [b, seq_len, ...]
        seq_2: [b, seq_len, ...]
        """
        
        seq_1v = seq_1[:, 1:, ...] - seq_1[:, :-1, ...]
        seq_2v = seq_2[:, 1:, ...] - seq_2[:, :-1, ...]

        return self.l1_loss(seq_1v, seq_2v)


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)
    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def forward(self, x, y):
        Gx = gram_matrix(x)
        Gy = gram_matrix(y)
        return F.mse_loss(Gx, Gy) * 30000000


class FrequencyLoss(nn.Module):
    def __init__(self, win_size=15):
        super().__init__()

    def get_frequency(self, x):
        fx = torch.fft.fft2(x, dim=(-2, -1))
        return fx
    
    def forward(self, x, y):
        fx = self.get_frequency(x)
        fy = self.get_frequency(y)

        return F.mse_loss(fx, fy) * 100


class FrequencyHighFreqLoss(nn.Module):
    def __init__(self, win_size=15, img_shape=(512, 512), mode='rec'):
        super().__init__()
        canvas = np.zeros(img_shape, dtype=np.uint8)
        center = (img_shape[0] // 2, img_shape[1] // 2)
        mask = cv2.circle(canvas, center, win_size, 255, -1)
        self.mask = np.transpose(mask[..., None], (2, 0, 1)) > 10
        self.mask = np.expand_dims(self.mask, 0)
        self.mode = mode

    def get_hf_frequency(self, x):
        fx_shift = []
        for i in range(x.shape[1]): # channel-wise
            _fx = torch.fft.fft2(x[:, i, ...][:, None, ...])
            _fx_shift = torch.fft.fftshift(_fx)
            b = x.shape[0]
            mask = np.stack([self.mask]*b, axis=0)
            _fx_shift[mask] = 0
            fx_shift.append(_fx_shift)

        return torch.cat(fx_shift, dim=1)
    
    def format_frequency(self, fx):
        return torch.log(torch.clamp(torch.abs(fx), min=0.00001))
    
    def recover_image(self, fx):
        i_x = []
        for i in range(fx.shape[1]): # channel-wise
            _i_fx = torch.fft.ifftshift(fx[:, i, ...][:, None, ...])
            _i_x = torch.fft.ifft2(_i_fx)
            i_x.append(_i_x)
        
        return torch.cat(i_x, dim=1)
    
    def forward(self, x, y):
        fx = self.get_hf_frequency(x)
        fy = self.get_hf_frequency(y)
        if self.mode == 'freq':
            fx = fx.real
            fy = fy.real
        if self.mode in ['format', 'rec'] :
            fx = self.format_frequency(fx)
            fy = self.format_frequency(fy)
        if self.mode == 'rec':
            fx = self.recover_image(fx).float()
            fy = self.recover_image(fy).float()
        
        # # safe check here
        # fx = torch.nan_to_num(fx, nan=0, posinf=10, neginf=0)
        # fy = torch.nan_to_num(fy, nan=0, posinf=10, neginf=0)

        return F.mse_loss(fx, fy) * 100


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, input, target, mask):        
        mask = mask.expand(-1, input.size()[1], -1, -1)
        loss = self.criterion(input * mask, target * mask)
        return loss


class TemporalPix2pixLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred_x, pred_fgr, true_x, true_fgr, true_mouth=None):
        assert len(pred_x.shape) == 5 and len(true_x.shape) == 5
        loss = dict()
        loss['l1'] = self.l1_loss(pred_x, true_x)
        m_pred_fgr = pred_x * true_fgr
        m_true_fgr = true_x * true_fgr
        loss['fgr_l1'] = self.l1_loss(m_pred_fgr, m_true_fgr) * 2

        loss['mask_l1'] = self.l1_loss(pred_fgr, true_fgr)

        loss['temp_coherence'] = F.mse_loss(pred_x[:, 1:] - pred_x[:, :-1],
                                            true_x[:, 1:] - true_x[:, :-1]) * 5
        loss['temp_fgr_coherence'] = F.mse_loss(pred_fgr[:, 1:] - pred_fgr[:, :-1],
                                            true_fgr[:, 1:] - true_fgr[:, :-1]) * 5
        
        loss['total'] = loss['l1'] + loss['fgr_l1'] + loss['mask_l1'] + loss['temp_coherence'] + loss['temp_fgr_coherence']

        if true_mouth is not None:
            mo_pred_fgr = pred_x * true_mouth
            mo_true_fgr = true_x * true_mouth
            loss['mouth_l1'] = self.l1_loss(mo_pred_fgr, mo_true_fgr) * 5
            loss['total'] = loss['total'] + loss['mouth_l1']
        
        return loss


from torchvision import models
class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out




