import torch
from models    import NICE
from loss      import StandardNormal, StandardLogistic
from utils     import load_checkpoint
from evaluate  import compute_test_loglikelihood, generate_samples

DATASET    = 'mnist'
CHECKPOINT = './checkpoints/ckpt_mnist_1500.pt'
PRIOR      = StandardLogistic() if DATASET in ('cifar10', 'svhn') else StandardNormal()

# Load the trained model
model = NICE.from_preset(DATASET)
load_checkpoint(CHECKPOINT, model, optimizer=None)
model.eval()

# 1. Report log-likelihood (paper comparison)
compute_test_loglikelihood(model, PRIOR, DATASET)

# 2. Generate and save sample images
generate_samples(model, PRIOR, DATASET, n_samples=100)