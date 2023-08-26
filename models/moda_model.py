import os
import torch
import numpy as np
from .componments import networks
from .base_model import BaseModel
from .componments import moda_net
from .componments.losses import MaskedMSELoss, MotionLoss


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

        self.loss_names = ['total', 'lipmotion', 'eyemovement', 'headmotion', 'torsomotion']

        if self.opt.loss == 'L2+Velocity':
            self.loss_names += ['lipmotion_vel', 'eyemovement_vel', 'headmotion_vel', 'torsomotion_vel']

        # losses
        self.featureL2loss = torch.nn.MSELoss().to(self.device)
        self.maskedL2loss = MaskedMSELoss().to(self.device)
        self.velocityloss = MotionLoss().to(self.device)

        if self.isTrain:
            # define only during training time
            # optimizer
            self.optimizer = torch.optim.AdamW([{'params': self.MODA.parameters(), 'initial_lr': opt.lr}], lr=opt.lr, betas=(0.9, 0.99))
            self.optimizers.append(self.optimizer)
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
            
            if opt.continue_train:
                self.resume_training()
    
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
    
    def resume_training(self):
        opt = self.opt
        ### if continue training, recover previous states            
        print('Resuming from epoch %s ' % (opt.load_epoch))   
        # change epoch count & update schedule settings
        opt.epoch_count = int(opt.load_epoch)
        self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        # print lerning rate
        lr = self.optimizers[0].param_groups[0]['lr']
        print('update learning rate: {} -> {}'.format(opt.lr, lr))

    def calculate_loss(self):
        """ calculate loss in detail, only forward pass included""" 
        self.loss = 0
        if 'L2' in self.opt.loss:
            self.loss_lipmotion   = self.featureL2loss(self.pred_lipmotion,   self.target_lipmotion)
            self.loss = self.loss + 5 * self.loss_lipmotion
            self.loss_eyemovement = self.featureL2loss(self.pred_eyemovement, self.target_eyemovement)
            self.loss_headmotion  = self.featureL2loss(self.pred_headmotion,  self.target_headmotion)
            self.loss_torsomotion = self.maskedL2loss(self.pred_torsomotion,  self.target_torsomotion, self.torsomask)
            self.loss = self.loss + self.loss_eyemovement + self.loss_headmotion + self.loss_torsomotion
        
        if 'Velocity' in self.opt.loss:
            self.loss_vel = 0
            self.loss_lipmotion_vel   = self.velocityloss(self.pred_lipmotion,   self.target_lipmotion)
            self.loss_vel = self.loss_vel + 5 * self.loss_lipmotion_vel
            self.loss_eyemovement_vel = self.velocityloss(self.pred_eyemovement, self.target_eyemovement)
            self.loss_headmotion_vel  = self.velocityloss(self.pred_headmotion,  self.target_headmotion)
            self.loss_torsomotion_vel = self.velocityloss(self.pred_torsomotion, self.target_torsomotion)
            self.loss_vel = self.loss_vel + self.loss_eyemovement_vel + self.loss_headmotion_vel + 0.5*self.loss_torsomotion_vel
            self.loss = self.loss + self.loss_vel
            
        self.loss_total = self.loss

    def set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        data contains: audio_samples, one_hot, audio_stride, target_point, target_headmotion
        """
        self.audio_feats, vertices_info, headmotion_info, torsomotion_info, torso_mask, _, meta  = data  #[:4]
  
        self.audio_feats = self.audio_feats.to(self.device)
        for k, v in meta.items():
            if k in ['Reference']: continue
            meta[k] = [x[0] for x in v]
        self.subject_id  = meta['Reference'].reshape(meta['Reference'].shape[0], -1).to(self.device)

        b, seq, t_dim = vertices_info.shape[:3]
        self.seq_len = seq
        vertices_info = vertices_info.reshape((b, seq, t_dim//3, 3))
        self.target_lipmotion   = vertices_info[:, :, meta['Lip']].reshape((b, seq, -1)).to(self.device)    
        self.target_eyemovement = vertices_info[:, :, meta['Eye']].reshape((b, seq, -1)).to(self.device)  
        self.target_headmotion  = headmotion_info.to(self.device)    
        self.target_torsomotion = torsomotion_info.to(self.device)
        self.torsomask = torso_mask.reshape((b, 1, 1)).to(self.device)
    
    def forward(self):

        pred_lip, pred_eye, pred_head, pred_torso = self.MODA(self.audio_feats, self.subject_id, frame_num=self.seq_len)[:4]

        self.pred_lipmotion   = pred_lip
        self.pred_eyemovement = pred_eye
        self.pred_headmotion  = pred_head
        self.pred_torsomotion = pred_torso

    
    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.calculate_loss()
        self.loss.backward()
        
    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        if self.audio_feats.shape[0] < 2: return        # skip batchnorm issue
        self.optimizer.zero_grad()   # clear optimizer parameters grad
        self.forward()               # forward pass
        self.backward()              # calculate loss and gradients
        self.optimizer.step()        # update gradients 
            
    @torch.no_grad()
    def validate(self):
        """ validate process """
        self.forward()
        self.calculate_loss()
    
    @torch.no_grad()
    def generate_sequences(self, audio_feats, sample_rate=16000, fps=30, n_subjects=165, sub_id=0, av_rate=534):

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
        audio_feats = audio_feats.to(self.device)
        while True:
            print('Current state:', i*step, i*step+win_size, audio_feats.shape[1])
            audio_feature_clip = audio_feats[:, i*step: i*step+win_size]
            
            frame_num = int(audio_feature_clip.shape[1] / av_rate)

            pred_lip, pred_eye, pred_head, pred_torso = self.MODA(audio_feature_clip, sub_info, frame_num=frame_num)[:4]
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

