import torch

COUNT_CLIENT = 6
SAMPLES_PER_CLIENT = 10000
ROOT = 'MNIST/dataset'
NOISE = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.0002
LABEL = 10
EPOCH = 100
THRESHOLD = 0.5
ROUNDS = 10
PROBABILITY = torch.tensor(0.7)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")