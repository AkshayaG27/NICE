# import torch
# from models    import NICE
# from loss      import StandardNormal, StandardLogistic
# from utils     import load_checkpoint
# from evaluate  import compute_test_loglikelihood, generate_samples

# DATASET    = 'mnist'
# CHECKPOINT = './checkpoints/ckpt_mnist_1500.pt'
# PRIOR      = StandardLogistic() if DATASET in ('cifar10', 'svhn') else StandardNormal()

# # Load the trained model
# model = NICE.from_preset(DATASET)
# load_checkpoint(CHECKPOINT, model, optimizer=None)
# model.eval()

# # 1. Report log-likelihood (paper comparison)
# compute_test_loglikelihood(model, PRIOR, DATASET)

# # 2. Generate and save sample images
# generate_samples(model, PRIOR, DATASET, n_samples=100)

# run_experiment.py

import torch
import os
import random
import numpy as np

from models    import NICE
from loss      import StandardNormal, StandardLogistic
from utils     import load_checkpoint
from evaluate  import compute_test_loglikelihood, generate_samples


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DATASET    = 'mnist'
CHECKPOINT = './checkpoints/ckpt_mnist_1500.pt'
N_SAMPLES  = 100
SEED       = 42


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ─────────────────────────────────────────────
# Safety checks
# ─────────────────────────────────────────────
if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")


# ─────────────────────────────────────────────
# Model + Prior
# ─────────────────────────────────────────────
model = NICE.from_preset(DATASET).to(device)

if DATASET in ('cifar10', 'svhn'):
    prior = StandardLogistic()
else:
    prior = StandardNormal()


# ─────────────────────────────────────────────
# Load checkpoint
# ─────────────────────────────────────────────
load_checkpoint(CHECKPOINT, model, optimizer=None)
model.eval()


# ─────────────────────────────────────────────
# Run evaluation
# ─────────────────────────────────────────────
print("\nEvaluating model...")

avg_ll, bpd = compute_test_loglikelihood(
    model,
    prior,
    DATASET
)


# ─────────────────────────────────────────────
# Generate samples
# ─────────────────────────────────────────────
print("\nGenerating samples...")

generate_samples(
    model,
    prior,
    DATASET,
    n_samples=N_SAMPLES
)


# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
print("\nFinal Summary")
print(f"Dataset: {DATASET}")
print(f"Log-likelihood: {avg_ll:.2f} nats")
print(f"BPD: {bpd:.4f}")
