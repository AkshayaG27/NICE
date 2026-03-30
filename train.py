# train.py

import os
import torch
import math
from models import NICE
from loss   import StandardNormal, StandardLogistic, nll_loss
from utils  import (dequantize, save_checkpoint, load_checkpoint,
                    get_dataloader, get_dataloader_valid)


def bits_per_dim(loss, dim):
    """
    Convert NLL to bits per dimension.
    """
    return loss / (dim * math.log(2))

import os

def get_checkpoint_dir():
    import os

    if os.path.exists('/content/drive/MyDrive'):
        checkpoint_dir = '/content/drive/MyDrive/NICE_checkpoints'
    else:
        checkpoint_dir = './checkpoints'

    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir
  
if __name__ == '__main__':

    # ── Config ────────────────────────────────────────────────────────
    DATASET        = 'mnist'
    EPOCHS         = 1500
    LR             = 2e-4
    BATCH_SIZE     = 200
    CHECKPOINT_DIR = get_checkpoint_dir()
    RESUME_FROM    = None
    CLIP_GRAD      = 5.0   # NEW

    PRIOR  = StandardLogistic() if DATASET in ('cifar10', 'svhn') else StandardNormal()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Dataset : {DATASET}")
    print(f"Device  : {device}")

    # ── Setup ─────────────────────────────────────────────────────────
    model        = NICE.from_preset(DATASET).to(device)
    loader       = get_dataloader(DATASET, batch_size=BATCH_SIZE)
    valid_loader = get_dataloader_valid(DATASET, batch_size=BATCH_SIZE)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, momentum=0.9)  #<- updated 0.0 to 0.9
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1/1.0005)

    start_epoch   = 0
    best_val_loss = float('inf')

    if RESUME_FROM:
        start_epoch = load_checkpoint(RESUME_FROM, model, optimizer)

    nvis = model.nvis

    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Initial LR : {scheduler.get_last_lr()[0]:.6f}")

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(start_epoch, EPOCHS):

        # ── TRAIN ────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0

        for x, _ in loader:
            x = x.view(x.size(0), -1)
            x = dequantize(x).to(device)

            loss = nll_loss(model, x, PRIOR)

            optimizer.zero_grad()
            loss.backward()

            # NEW: gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)

            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_train_loss = epoch_loss / len(loader)
        train_bpd = bits_per_dim(avg_train_loss, nvis)

        # # Momentum warmup
        # if epoch == 5:
        #     for param_group in optimizer.param_groups:
        #         param_group['momentum'] = 0.5
        #     print("  Momentum set to 0.5")              #<- removed this as its not sompatible with pytorch unlike pylearn2

        # ── VALIDATION ────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, _ in valid_loader:
                x = x.view(x.size(0), -1)
                x = dequantize(x).to(device)   # FIXED

                val_loss += nll_loss(model, x, PRIOR).item()

        avg_val_loss = val_loss / len(valid_loader)
        val_bpd = bits_per_dim(avg_val_loss, nvis)

        # ── DEBUG: latent statistics (VERY useful) ────────────────────
        with torch.no_grad():
            x_sample, _ = next(iter(loader))
            x_sample = dequantize(x_sample.view(x_sample.size(0), -1)).to(device)
            z = model(x_sample)

            z_mean = z.mean().item()
            z_std  = z.std().item()

        print(f"Epoch {epoch+1:4d}/{EPOCHS} | "
              f"train: {avg_train_loss:.4f} ({train_bpd:.4f} bpd) | "
              f"val: {avg_val_loss:.4f} ({val_bpd:.4f} bpd) | "
              f"z_mean: {z_mean:.3f} | z_std: {z_std:.3f} | "
              f"lr: {scheduler.get_last_lr()[0]:.6f}")

        # ── SAVE BEST ────────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(CHECKPOINT_DIR, f'best_{DATASET}.pt')
            save_checkpoint(model, optimizer, epoch + 1, best_path)
            print(f"  New best model at epoch {epoch+1} "
                  f"(val: {best_val_loss:.4f})")

        # ── REGULAR CHECKPOINT ───────────────────────────────────────
        if (epoch + 1) % 5 == 0:
            path = os.path.join(CHECKPOINT_DIR, f'ckpt_{DATASET}_{epoch+1}.pt')
            save_checkpoint(model, optimizer, epoch + 1, path)
