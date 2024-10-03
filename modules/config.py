import torch

# Configuration parameters
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Trasformer parameters
SOS_TOKEN = 0  # Start of sentence token
PKEEK = 0.5  # Probability of keeping a token

# Pretrained model parameters
PRETRAINED_MODEL_PARAMETERS_PATH = 'Insert path to pretrained model parameters'
PRETRAINED_MODEL_PATH = 'Insert path to pretrained model'

# GPT parameters
VOCAB_SIZE = 256
EMB_DIM = 512
NUM_HEADS = 16
NUM_LAYERS = 12
BLOCK_SIZE = 16*16
EMB_DROPOUT = 0.0
ATTENTION_DROPOUT = 0.0
RESIDUAL_DROPOUT = 0.0

# Saving parameters
SAVE_FOLDER = 'Insert path to save the model'
# Dataset parameters
IMG_DIR = 'Insert path to the dataset'
FRACTION = 0.003 # Fraction of the dataset to use
TRANSFORM = None # Transform to apply to the dataset
SUBPATCH = True

# Training parameters
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.1






