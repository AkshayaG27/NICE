# train.py

import os
import torch
from models import NICE
from loss   import StandardNormal, StandardLogistic, nll_loss
from utils  import (dequantize, save_checkpoint, load_checkpoint,
                    get_dataloader, get_dataloader_valid)  # ← add valid

if __name__ == '__main__':

    # ── Config ────────────────────────────────────────────────────────
    DATASET        = 'mnist'
    EPOCHS         = 1500
    LR             = 2e-4
    BATCH_SIZE     = 200
    CHECKPOINT_DIR = './checkpoints'
    RESUME_FROM    = None

    PRIOR  = StandardLogistic() if DATASET in ('cifar10', 'svhn') else StandardNormal()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Dataset : {DATASET}")
    print(f"Device  : {device}")

    # ── Setup ─────────────────────────────────────────────────────────
    model        = NICE.from_preset(DATASET).to(device)
    loader       = get_dataloader(DATASET, batch_size=BATCH_SIZE)
    valid_loader = get_dataloader_valid(DATASET, batch_size=BATCH_SIZE)  # ← add
    optimizer    = torch.optim.RMSprop(model.parameters(), lr=LR, momentum=0.0)
    scheduler    = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1/1.0005)

    start_epoch   = 0
    best_val_loss = float('inf')   # ← track best

    if RESUME_FROM:
        start_epoch = load_checkpoint(RESUME_FROM, model, optimizer)

    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Initial LR : {scheduler.get_last_lr()[0]:.6f}")

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(start_epoch, EPOCHS):

        # Training
        model.train()
        epoch_loss = 0.0
        for x, _ in loader:
            x = dequantize(x.view(x.size(0), -1)).to(device)
            loss = nll_loss(model, x, PRIOR)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()                  # ← once per epoch, outside batch loop
        avg_train_loss = epoch_loss / len(loader)

        # Momentum warmup — matches original MomentumAdjustor
        if epoch == 5:
            for param_group in optimizer.param_groups:
                param_group['momentum'] = 0.5
            print("  Momentum set to 0.5")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in valid_loader:
                x = x.view(x.size(0), -1).to(device)
                val_loss += nll_loss(model, x, PRIOR).item()
        avg_val_loss = val_loss / len(valid_loader)

        print(f"Epoch {epoch+1:4d}/{EPOCHS} | "
              f"train: {avg_train_loss:.4f} | "
              f"val: {avg_val_loss:.4f} | "
              f"lr: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(CHECKPOINT_DIR, f'best_{DATASET}.pt')
            save_checkpoint(model, optimizer, epoch + 1, best_path)
            print(f"  New best model at epoch {epoch+1} "
                  f"(val: {best_val_loss:.4f})")

        # Regular checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            path = os.path.join(CHECKPOINT_DIR, f'ckpt_{DATASET}_{epoch+1}.pt')
            save_checkpoint(model, optimizer, epoch + 1, path)
