import argparse
import os
from utils import util
import torch
import numpy as np
import models
from rich import print


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        ## task
        parser.add_argument('--task', type=str, default='Audio2FeatureFinetune', help='saving name.')
        
        ## basic parameters
        parser.add_argument('--model',           type=str, default='moda', help='trained model')
        parser.add_argument('--backbone',        type=str, default='Former', help='LSTM|Former')
        parser.add_argument('--dataset_mode',    type=str, default='audiovisual_v2', help='chooses how datasets are loaded. [audiovisual | audiovisual_v2 | audiovisual_v3], here v3 is for once, v2 is for single modality') 
        parser.add_argument('--name',            type=str, default='Audio2FeatureVertices', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids',         type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='repo/checkpoints/TalkingHead/Audio2Feature', help='models are saved here')  

        # dataset parameters
        parser.add_argument('--dataset_names',   type=str, default='HDTF')
        parser.add_argument('--dataroot',        type=str, default='repo/dataset/HDTF-semantic_mesh')
        parser.add_argument('--data_mode',       type=str, default='Train', help='|Train|Val|')
        parser.add_argument('--valid_part',      type=str, default='Lips', help='|Eyes|Lips|Iris|All')
        parser.add_argument('--num_threads',     default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size',      type=int, default=4, help='input batch size')
        parser.add_argument('--loss',            type=str, default='L2', help='|GMM|L2|')
        parser.add_argument('--sequence_length', type=int, default=240, help='length of training frames in each iteration')
        parser.add_argument('--serial_batches',  type=bool, default=False, help='Load sequence with order')
        
        # data setting parameters
        parser.add_argument('--FPS',               type=int, default=30, help='video fps')
        parser.add_argument('--sample_rate',       type=int, default=16000, help='audio sample rate')
        parser.add_argument('--rand_index',        type=bool, default=False, help='add rand perturbetion to dataloader')
        parser.add_argument('--use_norm',          type=bool, default=True, help='use normalized 3d features.')      # default is True
        parser.add_argument('--time_frame_length', type=int, default=600)
        parser.add_argument('--frame_jump_stride', type=int, default=300)
        parser.add_argument('--max_dataset_size',  type=int, default=5000)

        # model setting parameters
        parser.add_argument("--pretrained_faceformer",    type=str, default='audio2face/FaceFormer/vocaset/vocaset.pth')
        parser.add_argument('--extract_wav2vec',   type=bool, default=True,    help='Extract vector feature for audio, if not pre-computed. For dataset_v2')
        parser.add_argument('--audio_feat_dim',    type=int,  default=768,     help='feature dim for intermediate feature')
        parser.add_argument('--audio_head_type',   type=str,  default='wav2vec', choices=['wav2vec', 'apc', 'mel'])
        parser.add_argument('--feature_dim',       type=int,  default=64,      help='feature dim for intermediate feature')
        parser.add_argument('--hidden_size',       type=int,  default=128,     help='feature dim for intermediate feature')
        parser.add_argument('--vertice_dim',       type=int,  default=478*3,   help='output vertice dim for all face feature')
        parser.add_argument('--flame_vertice_dim', type=int,  default=1787*3,  help='output vertice dim for flame face feature')
        parser.add_argument('--lip_vertice_dim',   type=int,  default=40*3,    help='output vertice dim for lip feature')
        parser.add_argument('--lip_distill_vertice_dim', type=int, default=20*3,    help='output vertice dim for lip distill feature')
        parser.add_argument('--eye_vertice_dim',   type=int,  default=60*3,    help='output vertice dim for eye feature')
        parser.add_argument('--torso_vertice_dim', type=int,  default=18*3,    help='output vertice dim for shoulder feature')
        parser.add_argument('--max_seq_len',       type=int,  default=600,     help='max seq len for one iteration')
        parser.add_argument("--period",            type=int,  default=30,      help='length of one clip')
        parser.add_argument("--n_subjects",        type=int,  default=-1,      help='-1 means need to be updated from file/dataset')
        parser.add_argument("--motion_mode",       type=bool, default=False,   help='Switch model and network to pose estimation, like headpose, transformation')
        parser.add_argument("--tail_mode",         type=str,  default='All',   help='hyper-params for ablation in [All, LipMotion, Others]')
        parser.add_argument("--classifier_decoder",       type=bool, default=False,   help='Whether use classifier decoder')
        parser.add_argument('--teacher',           type=str,  default='config/teacher_net.yaml', help='teacher config path')
        
        # LSTM parameters
        parser.add_argument('--feature_decoder',  type=str, default='FormerHybrid', help='RNN|FormerHybrid|Identity')
        parser.add_argument('--body',  type=str,  default='base', help='base|dual|dualv2')
        parser.add_argument('--subject_head',     type=str, default='onehot', help='Ablation for subject head, in onehot|mesh|none')
        parser.add_argument('--decoder_tail',     type=str, default='verts', help='Used for mp2flame, [verts/params] for flame verts or params')
        parser.add_argument('--teacher_mode',     type=str, default='mouth_feature', help='inter_feature | mouth_feature | all_feature')
        parser.add_argument('--LSTM_input_size',  type=int, default=512)
        parser.add_argument('--LSTM_hidden_size', type=int, default=256)
        parser.add_argument('--LSTM_output_size', type=int, default=80)
        parser.add_argument('--LSTM_layers',      type=int, default=3)
        parser.add_argument('--LSTM_dropout',     type=float, default=0)
        parser.add_argument("--LSTM_residual", action="store_true")   
        parser.add_argument('--LSTM_sequence_length', type=int, default=60)

        # Wavenet parameters
        parser.add_argument('--A2L_wavenet_residual_layers',    type=int, default=7, help='residual layer numbers')
        parser.add_argument('--A2L_receptive_field',            type=int, default=10, help='receptive field for WavNet')
        parser.add_argument('--A2L_wavenet_residual_blocks',    type=int, default=2, help='residual block numbers')
        parser.add_argument('--A2L_wavenet_dilation_channels',  type=int, default=128, help='dilation convolution channels')
        parser.add_argument('--A2L_wavenet_residual_channels',  type=int, default=128, help='residual channels')
        parser.add_argument('--A2L_wavenet_skip_channels',      type=int, default=256, help='skip channels')      
        parser.add_argument('--A2L_wavenet_kernel_size',        type=int, default=2, help='dilation convolution kernel size')
        parser.add_argument('--A2L_wavenet_use_bias',           type=bool, default=True, help='whether to use bias in dilation convolution')
        parser.add_argument('--A2L_wavenet_cond',               type=bool, default=True, help='whether use condition input')
        parser.add_argument('--A2L_wavenet_cond_channels',      type=int, default=512, help='whether use condition input')
        parser.add_argument('--A2L_wavenet_input_channels',     type=int, default=12, help='input channels')        # 
        parser.add_argument('--A2L_GMM_ncenter',                type=int, default=1, help='gaussian distribution numbers, 1 for single gaussian distribution')
        parser.add_argument('--A2L_GMM_ndim',                   type=int, default=12, help='dimension of each gaussian, usually number of pts')
        parser.add_argument('--A2L_GMM_sigma_min',              type=float, default=0.03, help='minimal gaussian sigma values')
        parser.add_argument('--attention_d_model',              type=int, default=32, help='transformer d model')
        parser.add_argument('--attention_N',                    type=int, default=2, help='transformer N')
        parser.add_argument('--attention_n_heads',              type=int, default=2, help='n heads')
        
        # additional parameters
        parser.add_argument('--verbose',         action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--no_html',         default=True, help='no html for visualizer')
        parser.add_argument('--display_winsize', default=256, type=int, help='display winsize for visualizer')
        parser.add_argument('--suffix',          default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        

        self.initialized = True
        return parser
    

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        
        # save and return the parser
        self.parser = parser
        return opt

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        
        if opt.isTrain:
            # save to the disk
            expr_dir = os.path.join(opt.checkpoints_dir, opt.task, opt.name, opt.backbone)
            util.mkdirs(expr_dir)
            file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        if opt.motion_mode:
            print('##### In motion mode.')
            opt.vertice_dim = 7
            if opt.classifier_decoder:
                opt.vertice_dim = 256

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)

        self.opt = opt
        return self.opt











