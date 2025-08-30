import torch

# Data settings
DATA_DIR = "data"
IMAGE_DIR_NAME = "img"
KEY_DIR_NAME = "key"

# Model settings
BASE_MODEL_NAME = "naver-clova-ix/donut-base"
TRAINED_MODEL_DIR = "./trained_donut"

# Training settings
TRAIN_TEST_SPLIT = 0.1
MAX_LENGTH = 512
IMAGE_HEIGHT = 960
IMAGE_WIDTH = 720
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
NUM_TRAIN_EPOCHS = 30
LEARNING_RATE = 1e-5
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.05
EARLY_STOPPING_PATIENCE = 5

# Compute settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATALOADER_NUM_WORKERS = 4
