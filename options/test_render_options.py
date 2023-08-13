from .base_options_render import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--phase',              type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--dataset_type',       type=str, default='Test', help='')
        self.parser.add_argument('--use_example',        type=int, default=3, help='# of examples')
        self.parser.add_argument('--include_input',      action='store_true', default=True, help='if using positional encoding')
        self.parser.add_argument('--dataset_names',      type=str, default='WRA_CathyMcMorrisRodgers1_000', help='chooses test datasets.')
        self.parser.add_argument('--test_dataset_names', type=str, default='May', help='chooses validation datasets.')
        self.parser.add_argument('--load_epoch',         type=str, default='500', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--dynamic_example',             action='store_true', default=False, help='if using random crop for train')
        self.parser.add_argument('--dynamic_example_frame0_mode', type=int, default=1, help='0:Zeros; 1:Origin example; 2:Zeros+Origin example')
        self.parser.add_argument('--dynamic_example_loss_mode',   type=int, default=0, help='0:No loss; 1:add;')
        self.parser.add_argument('--use_position_encoding',       action='store_true', default=True, help='if using positional encoding')
        self.parser.add_argument('--position_encoding_size',      type=int, default=100, help='Value of position encoding')
        self.parser.add_argument('--num_encoding_functions',      type=int, default=6, help='Value of position encoding')
  
        self.isTrain = False
