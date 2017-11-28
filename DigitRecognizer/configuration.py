from easydict import EasyDict as edict
import os

# change me for different project directory
SYS_DIR = "/Users/xuan/dl_playground/DigitRecognizer"



__C = edict()
cfg = __C

# Define data dir configurations
__C.DIR = edict()

__C.DIR.data_dir = os.path.join(SYS_DIR, "data")
__C.DIR.training_data = os.path.join(__C.DIR.data_dir, "MNIST/train.csv")
__C.DIR.testing_data = os.path.join(__C.DIR.data_dir, "MNIST/test.csv")

__C.DIR.model_save_dir = os.path.join(SYS_DIR, "model_checkpoints")

__C.DIR.tensorboard_log_dir = os.path.join(SYS_DIR, "tensorboard")

# HyperParameters for Training.
__C.TRAIN = edict()
__C.TRAIN.LEARNING_RATE = 0.1
__C.TRAIN.LEARNING_RATE_STEP_SIZE = 1000
__C.TRAIN.LEARNING_RATE_DECAY_RATE = 0.95
__C.TRAIN.BATCH_SIZE = 5
__C.TRAIN.EPOCHS = 100
__C.TRAIN.SAVE_STEPS = 25
__C.TRAIN.DATA_SPLIT_RATIO = 0.2
__C.TRAIN.VALIDATE_EPOCHES = 50

__C.TEST = edict()
__C.TEST.BATCH_SIZE = 5