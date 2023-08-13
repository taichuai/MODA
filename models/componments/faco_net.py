import torch
import torch.nn as nn
from .networks import MLP


class FaCoNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        in_m_dim = opt.lip_vertice_dim 
        in_e_dim = opt.eye_vertice_dim
        out_dim = opt.vertice_dim       # - in_dim
        
        if opt.subject_head == 'onehot':
            self.subject_encoder = MLP(opt.n_subjects, 256, 256, 3)
        elif opt.subject_head == 'point':
            self.subject_encoder = MLP(opt.vertice_dim, 256, 128, 3)
        
        self.driven_m_encoder = MLP(in_m_dim, 128, 256, 3)
        self.driven_e_encoder = MLP(in_e_dim, 128, 256, 3)

        self.decoder = MLP(256, out_dim, 512, 4)
                    
    
    def forward(self, x_in_lip, x_in_eye, ref):
        '''
        Args:
            x_in: [b, T, ndim]  Driving vertices
        '''
        bs, item_len, ndim_lip = x_in_lip.shape
        bs, item_len, ndim_eye = x_in_eye.shape
        
        x_m_driven = self.driven_m_encoder(x_in_lip.reshape(-1, ndim_lip)).reshape(bs, item_len, -1)
        x_e_driven = self.driven_e_encoder(x_in_eye.reshape(-1, ndim_eye)).reshape(bs, item_len, -1)
        
        x_driven = torch.concat([x_m_driven, x_e_driven], dim=-1)
        x_suj    = self.subject_encoder(ref).unsqueeze(1)

        y_pred = self.decoder((x_driven + x_suj).reshape(-1, 256)).reshape(bs, item_len, -1)
        
        return y_pred



class PointDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PointDiscriminator, self).__init__()
        self.opt = opt
        in_dim = opt.vertice_dim
        out_dim = opt.lip_vertice_dim + opt.eye_vertice_dim
        
        if opt.subject_head == 'onehot':
            self.subject_encoder = MLP(opt.n_subjects, 256, 256, 3)
        elif opt.subject_head == 'mesh':
            self.subject_encoder = MLP(opt.vertice_dim, 256, 128, 3)
        self.mesh_encoder = MLP(in_dim+out_dim, 256, 256, 2)

        self.classifier = MLP(256, 1, 128, 3)
                    
    
    def forward(self, x_in, onehot):
        '''
        Args:
            x_in: [b, T, ndim]  Driving vertices
        '''
        bs, item_len, ndim = x_in.shape
        x_driven = self.mesh_encoder(x_in.reshape(-1, ndim)).reshape(bs, item_len, -1)
        x_suj    = self.subject_encoder(onehot).unsqueeze(1)

        y_pred = self.classifier((x_driven + x_suj).reshape(-1, 256)).reshape(bs, item_len, -1)
        
        return y_pred

