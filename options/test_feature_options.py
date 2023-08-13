from .base_options_feature import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--phase',      type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--load_epoch', type=str, default='500', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--eval',       action='store_true', help='use eval mode during test time.')
        parser.set_defaults(time_frame_length=1)
        self.isTrain = False
        
        return parser
