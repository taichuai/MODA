import os
import torch
import numpy as np
from .componments import networks
from .base_model import BaseModel
from .componments import moda_net


class MODAModel(BaseModel):
    def __init__(self, opt):
        """Initialize the MODA Model class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.Tensor = torch.cuda.FloatTensor
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        # define networks 
        self.model_names = ['MODA']
        self.MODA = networks.init_net(moda_net.MODANet(opt), init_type='normal', init_gain=0.02, gpu_ids=opt.gpu_ids)
    
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
        return self.MODA(audio_array, sub_info, frame_num)
    
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

