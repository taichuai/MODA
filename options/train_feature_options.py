from .base_options_feature import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # network saving and loading parameters
        parser.add_argument('--save_epoch_freq', type=int           , default=10                              , help='frequency of saving checkpoints at the end of epochs'), 
        parser.add_argument('--save_by_iter'   , action='store_true', help='whether saves model by iteration'),
        parser.add_argument('--continue_train' , default=False      , action='store_true'                     , help='continue training: load the latest model'),
        parser.add_argument('--load_epoch'     , type=str           , default='200'                           , help='which epoch to load? set to latest to use latest cached model'),
        parser.add_argument('--epoch_count'    , type=int           , default=0                               , help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase'          , type=str           , default='train'                         , help='train, val, test, etc')
        parser.add_argument('--re_transform'   , type=int           , default=0                               , help='re-transform landmarks'),
        
        # training parameters
        parser.add_argument('--train_dataset_names'   , type=str  , default='train_list.txt',       help='chooses train datasets.'),
        parser.add_argument('--validate_dataset_names', type=str  , default='val_list.txt'  ,       help='chooses validation datasets.'),
        parser.add_argument('--subject_name'          , type=str  , default='May, Obama'    ,       help='chooses train datasets.'),
        parser.add_argument('--train_sub_names'       , type=str  , default='May, Obama'    ,       help='chooses train datasets.'),
        parser.add_argument('--test_sub_names'        , type=str  , default='May-test, Obama-test', help='chooses validation datasets.'),
        parser.add_argument('--ref_sub_names'         , type=str  , default='May, Obama'    , help='chooses reference datasets.') ,
        parser.add_argument('--dataset_type'          , type=str  , default='Train'         , help='[Train or Test]')        ,
        parser.add_argument('--n_epochs'              , type=int  , default=1000            , help='number of epochs')       ,
        parser.add_argument('--lr_policy'             , type=str  , default='cosine'        , help='learning rate policy. [linear | step | plateau | cosine]'),
        parser.add_argument('--lr'                    , type=float, default=1e-3            , help='initial learning rate for adam'),
        parser.add_argument('--gamma'                 , type=float, default=0.2             , help='step learning rate gamma'),
        parser.add_argument('--lr_decay_iters'        , type=int  , default=250             , help='multiply by a gamma every lr_decay_iters iterations'),
        parser.add_argument('--n_epochs_decay'        , type=int  , default=250             , help='number of epochs to linearly decay learning rate to zero'),
        parser.add_argument('--validate_epoch'        , type=int  , default=50              , help='validate model every some epochs, 0 for not validate during training'),
        parser.add_argument('--loss_smooth_weight'    , type=float, default=0               , help='smooth loss weight, 0 for not use smooth loss'),
        parser.add_argument('--weight_decay'          , type=float, default=1e-5            , help='weight decay'),
        parser.add_argument('--optimizer'             , type=str  , default='AdamW'         , help='SGD, Adam, AdamW, RMSprop')
                
        # data augmentations
        parser.add_argument('--gaussian_noise'      , type=int  , default=0   , help='whether add gaussian noise to input & groundtruth features')
        parser.add_argument('--gaussian_noise_scale', type=float, default=0.01, help='gaussian noise scale')
        
        parser.add_argument('--TTUR'     , action='store_true', help='Use TTUR training scheme'),
        parser.add_argument('--gan_mode' , type=str           , default='ls'                    , help='(ls|original|hinge)')
        parser.add_argument('--beta1'    , type=float         , default=0.5                     , help='momentum term of adam')
        parser.add_argument('--lambda_L1', type=float         , default=100.0                   , help='weight for temporal loss')
        
        # for discriminators 
        parser.add_argument('--no_discriminator', type=int           , default=0                                                , help='not use discriminator')
        parser.add_argument('--num_D'           , type=int           , default=2                                                , help='number of patch scales in each discriminator')
        parser.add_argument('--n_layers_D'      , type=int           , default=3                                                , help='number of layers in discriminator')
        parser.add_argument('--no_ganFeat'      , action='store_true', help='do not match discriminator features')              ,
        parser.add_argument('--lambda_feat'     , type=float         , default=10.0                                             , help='weight for feature matching')
        parser.add_argument('--sparse_D'        , action='store_true', help='use sparse temporal discriminators to save memory'),

        self.isTrain = True
        return parser



