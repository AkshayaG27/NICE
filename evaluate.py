# evaluate.py

import torch
import torchvision
import matplotlib.pyplot as plt
import os
from models import NICE
from loss   import StandardNormal, StandardLogistic, nll_loss
from utils  import get_dataloader, load_checkpoint


def compute_test_loglikelihood(model, prior, dataset_name, batch_size=200):
    """
    Runs the model on the test set and reports average log-likelihood in nats.
    This is the number the paper reports in Table 1.
    """
    model.eval()

    # Get the test split (not train)
    from utils import get_dataloader_test
    loader = get_dataloader_test(dataset_name, batch_size)

    total_log_prob = 0.0
    total_samples  = 0

    with torch.no_grad():
        for x, _ in loader:
            x = x.view(x.size(0), -1)

            # No dequantization at test time — we evaluate on clean data
            z      = model.encode(x)
            log_pz = prior.log_prob(z)
            ldj    = model.log_det_jacobian()
            log_px = (log_pz + ldj)          # shape: (batch,)

            total_log_prob += log_px.sum().item()
            total_samples  += x.size(0)

    avg_log_likelihood = total_log_prob / total_samples
    print(f"Test log-likelihood: {avg_log_likelihood:.2f} nats")
    print(f"(Paper reports: MNIST=-1454, CIFAR=-5371, SVHN=-5853, TFD=-4483)")
    return avg_log_likelihood


def generate_samples(model, prior, dataset_name, n_samples=100):
    """
    Generates new images by:
      1. Sampling z from the prior (random noise)
      2. Running z backwards through the model (decode)
      3. Saving as an image grid
    """
    model.eval()

    nvis = NICE.PRESETS[dataset_name]['nvis']

    with torch.no_grad():
        # Step 1: sample from prior
        z = prior.sample(n_samples, nvis)

        # Step 2: decode z → x
        x = model.decode(z)

    # Step 3: reshape flat vectors back to images for display
    if dataset_name == 'mnist':
        x = x.view(n_samples, 1, 28, 28)
        x = x.clamp(0, 1)
    elif dataset_name in ('cifar10', 'svhn'):
        x = x.view(n_samples, 3, 32, 32)
        x = (x * 0.5 + 0.5).clamp(0, 1)  # undo the [-1,1] normalization
    elif dataset_name == 'tfd':
        x = x.view(n_samples, 1, 48, 48)
        x = x.clamp(0, 1)

    # Save as a grid image
    os.makedirs('./outputs', exist_ok=True)
    grid_path = f'./outputs/samples_{dataset_name}.png'
    torchvision.utils.save_image(x, grid_path, nrow=10, padding=2)
    print(f"Saved {n_samples} samples → {grid_path}")

    return x