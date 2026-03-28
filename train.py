# train.py

import os
import torch
from models import NICE
from loss   import StandardNormal, StandardLogistic, nll_loss
from utils  import dequantize, save_checkpoint, load_checkpoint, get_dataloader

# ── Config ──────────────────────────────────────────────────────────────────
DATASET       = 'mnist'    # change to 'cifar10', 'svhn', or 'tfd'
EPOCHS        = 3
LR            = 2e-4
BATCH_SIZE    = 200
CHECKPOINT_DIR = './checkpoints'
RESUME_FROM   = None       # set to a path like 'checkpoints/ckpt_mnist_100.pt' to resume

# Use logistic prior for cifar10/svhn, normal for mnist/tfd
PRIOR = StandardLogistic() if DATASET in ('cifar10', 'svhn') else StandardNormal()

# ── Setup ───────────────────────────────────────────────────────────────────
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

model     = NICE.from_preset(DATASET)
loader    = get_dataloader(DATASET, batch_size=BATCH_SIZE)
optimizer = torch.optim.RMSprop(model.parameters(), lr=LR)

# LR decay: multiply LR by (1/1.0005) every step, matching the original yaml
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1/1.0005)

start_epoch = 0
if RESUME_FROM:
    start_epoch = load_checkpoint(RESUME_FROM, model, optimizer)

# ── Training loop ───────────────────────────────────────────────────────────
for epoch in range(start_epoch, EPOCHS):
    epoch_loss = 0.0

    for x, _ in loader:
        x = dequantize(x.view(x.size(0), -1))

        loss = nll_loss(model, x, PRIOR)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()           # step per batch, matching original yaml

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    print(f"Epoch {epoch+1:4d}/{EPOCHS} | loss: {avg_loss:.4f} | lr: {scheduler.get_last_lr()[0]:.6f}")

    if (epoch + 1) % 10 == 0:
        path = os.path.join(CHECKPOINT_DIR, f'ckpt_{DATASET}_{epoch+1}.pt')
        save_checkpoint(model, optimizer, epoch + 1, path)