import math
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
EPOCH = 1 ## ^0^
EPSILON = 0.001
DELTA = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## Formula taken from DPGAN article
## https://arxiv.org/pdf/1802.06739
## σ = 2q[√n_d*log(1/δ)]/ɛ
SIGMA = 2*(BATCH_SIZE/60_000)*(math.sqrt(CRITIC_ITER*math.log10(1/DELTA))/EPSILON)