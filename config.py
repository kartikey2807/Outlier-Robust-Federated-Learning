import torch
import torch.nn as nn

MAX_BYZANTINE = 0
COUNT_CLIENT  = 6
SAMPLE_LEN = 10000
ROOT = 'MNIST/dataset'
THRESHOLD = 0.5
ROUNDS = 30
NOISE = 100
BATCH_SIZE = 128
CRITIC_ITER = 5
CLIP = 0.01
LEARNING_RATE = 0.0005
EPOCH = 40 ## x10
EPSILON = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")