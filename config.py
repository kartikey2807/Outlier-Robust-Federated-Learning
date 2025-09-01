import torch
import torch.nn as nn

MAX_BYZANTINE = 2
COUNT_CLIENT  = 6
SAMPLES_PER_CLIENT = 10000
ROOT = 'MNIST/dataset'
THRESHOLD = 0.5
ROUNDS = 10
NOISE = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.0002
LABEL = 10
EPOCH = 100
closest_neighbor = COUNT_CLIENT - MAX_BYZANTINE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")