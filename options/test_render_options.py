from .base_options_render import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--phase',              type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--dataset_type',       type=str, default='Test', help='')
        self.parser.add_argument('--dataset_names',      type=str, default='WRA_CathyMcMorrisRodgers1_000', help='chooses test datasets.')
        self.parser.add_argument('--test_dataset_names', type=str, default='May', help='chooses validation datasets.')
        self.parser.add_argument('--load_epoch',         type=str, default='500', help='which epoch to load? set to latest to use latest cached model')
  
        self.isTrain = False
