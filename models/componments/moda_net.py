
import torch
import numpy as np
import torch.nn as nn
from .wav2vec import Wav2Vec2Model
from .networks import MLP, DualTemporalMoudleV2


class MODANet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.input_size = opt.audio_feat_dim
        self.audio_head_type = opt.audio_head_type
        
        self.hs = opt.hidden_size        # larger parameters
        if self.audio_head_type == 'wav2vec':
            self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            self.audio_encoder.feature_extractor._freeze_parameters()
            self.audio_encoder_head = MLP(opt.audio_feat_dim, opt.hidden_size, opt.hidden_size, 2, with_norm=True)
        elif self.audio_head_type == 'apc':
            self.audio_encoder_head = MLP(512 * 4, opt.hidden_size, opt.hidden_size, 2, with_norm=True)
        elif self.audio_head_type == 'mel':
            self.audio_encoder_head = MLP(80 * 4,  opt.hidden_size, opt.hidden_size, 4, with_norm=True)
        else:
            raise ValueError
        self.subject_encoder_head = MLP(opt.vertice_dim, opt.hidden_size, 128, 3, with_norm=True)

        self.temporal_body    = DualTemporalMoudleV2(opt.hidden_size, self.hs, 3, 0, opt.feature_decoder, opt.period, opt.max_seq_len)

        hs = self.hs*2
        self.lipmotion_tail   = MLP(hs, opt.lip_vertice_dim,   512, 2, with_norm=True)
        self.eyemovement_tail = MLP(hs, opt.eye_vertice_dim,   256, 3, with_norm=True)
        self.headmotion_tail  = MLP(hs, 3+3+1,                 256, 3, with_norm=True)
        self.torsomotion_tail = MLP(hs, opt.torso_vertice_dim, 256, 3, with_norm=True)
    
    def forward(self, audio_array, sub_info, frame_num=None):
        '''
        Args:
            audio_features: [b, T, ndim]
        '''
        if self.audio_head_type == 'wav2vec':
            audio_features = self.audio_encoder(audio_array, frame_num=frame_num).last_hidden_state
            down_audio_feats = self.audio_encoder_head(audio_features)
        elif self.audio_head_type in ('apc', 'mel'):
            bs, item_len, ndim = audio_array.shape
            # 120 fps -> 30 fps
            audio_features = audio_array.reshape(bs, -1, ndim*4)
            down_audio_feats = self.audio_encoder_head(audio_features.reshape(-1, ndim*4)).reshape(bs, int(item_len/4), -1)
            item_len = down_audio_feats.shape[1]
        else:
            raise ValueError

        subject_style = self.subject_encoder_head(sub_info)

        output_feat, mu, logvar = self.temporal_body(down_audio_feats, subject_style)
        
        pred_lipmotion   = self.lipmotion_tail(output_feat)
        pred_eyemovement = self.eyemovement_tail(output_feat)
        pred_headmotion  = self.headmotion_tail(output_feat)    
        pred_torsomotion = self.torsomotion_tail(output_feat)
        
        return pred_lipmotion, pred_eyemovement, pred_headmotion, pred_torsomotion, (mu, logvar)

