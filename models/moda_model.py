
import torch
import numpy as np
import torch.nn as nn
from .base_model import BaseModel
from .componments.wav2vec import Wav2Vec2Model
from .componments.networks import MLP, DualTemporalMoudleV2


class MODAModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        
        self.input_size = opt.audio_feat_dim
        self.audio_head_type = opt.audio_head_type
        
        self.hs = opt.hidden_size        # larger parameters
        if self.audio_head_type == 'wav2vec':
            self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            self.audio_encoder.feature_extractor._freeze_parameters()
            self.audio_encoder.eval()
            self.audio_encoder_head = MLP(opt.audio_feat_dim, opt.hidden_size, opt.hidden_size, 2)
        elif self.audio_head_type == 'apc':
            self.audio_encoder_head = MLP(512 * 4, opt.hidden_size, opt.hidden_size, 2, with_norm=True)
        elif self.audio_head_type == 'mel':
            self.audio_encoder_head = MLP(80 * 4,  opt.hidden_size, opt.hidden_size, 4, with_norm=True)
        else:
            raise ValueError
        self.subject_encoder_head = MLP(opt.vertice_dim, opt.hidden_size, 128, 3, with_norm=True)

        self.temporal_body    = DualTemporalMoudleV2(opt.hidden_size, self.hs, 3, 0, opt.feature_decoder, opt.period, opt.max_seq_len)

        hs = self.hs*2 if opt.feature_decoder == 'FormerHybrid' else self.hs
        self.lipmotion_tail   = MLP(hs, opt.lip_vertice_dim,   512, 2, with_norm=True)
        self.eyemovement_tail = MLP(hs, opt.eye_vertice_dim,   256, 3, with_norm=True)
        self.headmotion_tail  = MLP(hs, 3+3+1,                 256, 3, with_norm=True)
        self.torsomotion_tail = MLP(hs, opt.torso_vertice_dim, 256, 3, with_norm=True)
    
    def set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        data contains: audio_samples, one_hot, audio_stride, target_point, target_headmotion
        """
        self.audio_feats, vertices_info, headmotion_info, torsomotion_info, torso_mask, meta  = data  #[:4]
        if self.motion_mode:
            self.target_info = data[-1]
        self.audio_feats = self.audio_feats.to(self.device)
        for k, v in meta.items():
            if k in ['MeanPoint']: continue
            meta[k] = [x[0] for x in v]
        if self.opt.subject_head == 'onehot':
            self.subject_id  = self.one_hot.to(self.device)
        else:
            self.subject_id  = meta['MeanPoint'].reshape(meta['MeanPoint'].shape[0], -1).to(self.device)

        b, seq, t_dim = vertices_info.shape[:3]
        vertices_info = vertices_info.reshape((b, seq, t_dim//3, 3))
        self.target_lipmotion   = vertices_info[:, :, meta['Lip']].reshape((b, seq, -1)).to(self.device)    
        self.target_eyemovement = vertices_info[:, :, meta['Eye']].reshape((b, seq, -1)).to(self.device)  
        self.target_headmotion  = headmotion_info.to(self.device)    
        self.target_torsomotion = torsomotion_info.to(self.device)
        self.torsomask = torso_mask.reshape((b, 1, 1)).to(self.device)
    
    def forward(self, audio_array, sub_info, frame_num=None):
        '''
        Args:
            audio_features: [b, T, ndim]
        '''
        if self.audio_head_type == 'wav2vec':
            audio_features = self.audio_encoder(audio_array, frame_num=frame_num).last_hidden_state.detach()
            bs, item_len, ndim = audio_features.shape
            down_audio_feats = self.audio_encoder_head(audio_features.reshape(-1, ndim)).reshape(bs, item_len, -1)
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
    
    @torch.no_grad()
    def generate_sequences(self, audio_feats, sample_rate=16000, fps=30, n_subjects=165, sub_id=0, av_rate=534):

        self.subject_encoder_head.eval()
        step = sample_rate*10
        win_size = step*2
        predictions_lipmotion   = []
        predictions_eyemovement = []
        predictions_headmotion  = []
        predictions_torsomotion = []
        if self.opt.subject_head == 'onehot':
            one_hot_labels = np.eye(n_subjects)
            one_hot = one_hot_labels[sub_id]
            one_hot = np.reshape(one_hot,(-1,one_hot.shape[0]))
            one_hot = torch.FloatTensor(one_hot).to(device=self.device)
            sub_info = one_hot
        elif self.opt.subject_head == 'point':
            t_sub = torch.from_numpy(sub_id)
            sub_info = t_sub.unsqueeze(0).float()
            sub_info = sub_info.reshape(sub_info.shape[0], -1).to(self.device)
            # print(sub_info.shape)
        
        i = 0
        while True:
            print('Current state:', i*step, i*step+win_size, audio_feats.shape[1])
            audio_feature_clip = audio_feats[:, i*step: i*step+win_size]
            
            frame_num = int(audio_feature_clip.shape[1] / av_rate)

            pred_lip, pred_eye, pred_head, pred_torso = self.forward(audio_feature_clip, sub_info, frame_num=frame_num)[:4]
            predictions_lipmotion.append(pred_lip.squeeze())
            predictions_eyemovement.append(pred_eye.squeeze())
            predictions_headmotion.append(pred_head.squeeze())
            predictions_torsomotion.append(pred_torso.squeeze())
            if i*step+win_size > audio_feats.shape[1]:
                break
            i+=1
        
        if len(predictions_lipmotion) > 1:
            # merge predictions
            prediction_lipmotion   = predictions_lipmotion[0]
            prediction_eyemovement = predictions_eyemovement[0]
            prediction_headmotion  = predictions_headmotion[0]
            prediction_torsomotion = predictions_torsomotion[0]
            mid_len = prediction_lipmotion.shape[0] // 2
            for i in range(len(predictions_lipmotion) - 1):
                next_prediction_lipmotion = predictions_lipmotion[i+1]
                prediction_lipmotion[-mid_len:, :] = (prediction_lipmotion[-mid_len:, :] + next_prediction_lipmotion[:mid_len, :]) / 2.
                prediction_lipmotion = torch.cat([prediction_lipmotion, next_prediction_lipmotion[mid_len:, :]], dim=0)
                
                next_prediction_eyemovement = predictions_eyemovement[i+1]
                prediction_eyemovement[-mid_len:, :] = (prediction_eyemovement[-mid_len:, :] + next_prediction_eyemovement[:mid_len, :]) / 2.
                prediction_eyemovement = torch.cat([prediction_eyemovement, next_prediction_eyemovement[mid_len:, :]], dim=0)
                
                next_prediction_headmotion = predictions_headmotion[i+1]
                prediction_headmotion[-mid_len:, :] = (prediction_headmotion[-mid_len:, :] + next_prediction_headmotion[:mid_len, :]) / 2.
                prediction_headmotion = torch.cat([prediction_headmotion, next_prediction_headmotion[mid_len:, :]], dim=0)
                
                next_prediction_torsomotion = predictions_torsomotion[i+1]
                prediction_torsomotion[-mid_len:, :] = (prediction_torsomotion[-mid_len:, :] + next_prediction_torsomotion[:mid_len, :]) / 2.
                prediction_torsomotion = torch.cat([prediction_torsomotion, next_prediction_torsomotion[mid_len:, :]], dim=0)

        else:
            prediction_lipmotion   = predictions_lipmotion[0]
            prediction_eyemovement = predictions_eyemovement[0]
            prediction_headmotion  = predictions_headmotion[0]
            prediction_torsomotion = predictions_torsomotion[0]

        preds_lip   = prediction_lipmotion.detach().cpu().numpy()
        preds_eye   = prediction_eyemovement.detach().cpu().numpy()
        preds_head  = prediction_headmotion.detach().cpu().numpy()
        preds_torso = prediction_torsomotion.detach().cpu().numpy()
            
        return preds_lip, preds_eye, preds_head, preds_torso

