import torch
import torch.nn as nn

MAX_BYZANTINE = 3
COUNT_CLIENT  = 6
SAMPLE_LEN = 10000
ROOT = 'MNIST/dataset'
THRESHOLD = 0.5
ROUNDS = 40
NOISE = 100
BATCH_SIZE = 64
CRITIC_ITER = 5
CLIP = 0.01
LEARNING_RATE_C = 0.0005
LEARNING_RATE_G = 0.00005
EPOCH = 30 ## x10
EPSILON = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")