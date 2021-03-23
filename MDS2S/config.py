import collections
import numpy as np


class Config(object):

    MDS = 'seq2seq'  # seq2seq or transformer

    BATCH_SIZE = 8

    CHANNELS = [0, 1, 2]

    # Data collective params
    SAMPLING_RATE = 24e3
    SAMPLING_LENGTH = 1e4
    T = np.arange(0, 1e4/SAMPLING_RATE, 1.0/SAMPLING_RATE)

    # Data address
    DATA_ROOT_DIR = "/Users/zcc/Documents/wield/data"
    LABEL_ROOT_DIR = "/Users/zcc/Documents/wield/Loc&Depth"
    IMFs_ROOT_DIR = "/Users/zcc/Documents/wield/IMFs"
    FINAL_ROOT_DIR = "/Users/zcc/Documents/wield/final"

    # Preprocess
    TIME_RANGE = [700, 3500]  # total = 52 x 52 = 2704
    FREQ_RANGE = [2626, -29]  # total = 144
    SLICE_LENGTH = 144
    SCALE = 1e2

    # Wavelet TF params
    W_TF = "cgau8"
    TOTALSCALE = TIME_RANGE[1] - TIME_RANGE[0]

    # STFT params
    WINDOW = 95
    STEP_SIZE = 96
    USE_STFT = False

    AMPLIFIER = 1

    # The localization area
    LOCAL_HEIGHT = 80
    LOCAL_WIDTH = 110

    # Name of the model
    NAME = None

    # Seq2seq hyper parameters
    ENCODER_BACKBONE = 'efficientnet-b3'
    DECODER_BACKBONE = 'custom'

    ENCODER_REPEAT = 5
    DECODER_REPEAT = 5
    DCM_REPEAT = 6

    GPU_COUNT = 1
    SIGNAL_PIC_PER_IMAGE = 1

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    VERSION = 0.1

    LOSS_WEIGHTS = {'custom': 100,
                    'transformer': 1}

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = False  # Defaulting to False since batch size is often small

    # Gradient norm clipping, used in optimizer setups
    GRADIENT_CLIP_NORM = 5.0

    # MBConv args
    BlockArgs = collections.namedtuple('BlockArgs', [
        'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
        'expand_ratio', 'id_skip', 'strides', 'se_ratio'
    ])
    # defaults will be a public argument for namedtuple in Python 3.7
    # https://docs.python.org/3/library/collections.html#collections.namedtuple
    BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

    DEFAULT_BLOCKS_ARGS = [
        BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
                  expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
        BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
                  expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
        BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
                  expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
        BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
                  expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
        BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
                  expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
        BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
                  expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
        BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
                  expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
    ]

    CONV_KERNEL_INITIALIZER = {
        'class_name': 'VarianceScaling',
        'config': {
            'scale': 2.0,
            'mode': 'fan_out',
            # EfficientNet actually uses an untruncated normal distribution for
            # initializing conv layers, but keras.initializers.VarianceScaling use
            # a truncated distribution.
            # We decided against a custom initializer for better serializability.
            'distribution': 'normal'
        }
    }

    TOP_DOWN_PYRAMID_SIZE = 256

    # BLOCK_NUMS: for drop rate calculation.
    BLOCKS_NUMS = 0

    DROP_CONNECT_RATE = 0.2

    # TODO: Add damage type dictionary.

    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.SIGNAL_PIC_PER_IMAGE * self.GPU_COUNT
        self.SHUFFLE = True

        # TOTAL_BLOCK_NUMS: for drop rate calculation.
        self.TOTAL_BLOCK_NUMS = sum(block_args.num_repeat for block_args in self.DEFAULT_BLOCKS_ARGS)

        # self.computeDecoderInputResolution()

    def calculateDropRate(self, block_args):
        pass

    # def computeDecoderInputResolution(self):
    #
    #     self.resolution = 9 * self.SIGNAL_FREQ // 8
    #     self.DECODER_INPUT_SHAPE = [None, self.resolution, self.resolution, self.TOP_DOWN_PYRAMID_SIZE]

    def to_dict(self):
        return {a: getattr(self, a)
                for a in sorted(dir(self))
                if not a.startswith("__") and not callable(getattr(self, a))}

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for key, val in self.to_dict().items():
            print(f"{key:30} {val}")
        # for a in dir(self):
        #     if not a.startswith("__") and not callable(getattr(self, a)):
        #         print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


if __name__ == '__main__':
    c = Config()
    # print(c.resolution)
