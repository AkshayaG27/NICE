import matplotlib.pyplot as plt
import torchvision.utils as vutils

def plot_loss(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log-likelihood)")
    plt.title("Training Curve")
    plt.legend()
    plt.grid()
    plt.show()

def show_real_vs_generated(model, prior, data_loader, device, n=64):
    model.eval()

    # ---- Get REAL images ----
    x_real, _ = next(iter(data_loader))
    x_real = x_real[:n].to(device)

    # ---- Generate FAKE images ----
    with torch.no_grad():
        z = prior.sample((n,)).to(device)
        x_fake = model.inverse(z)
        x_fake = torch.sigmoid(x_fake)  # if needed

    # ---- Make grids ----
    real_grid = vutils.make_grid(
        x_real.view(-1, 1, 28, 28), nrow=8, normalize=True
    )

    fake_grid = vutils.make_grid(
        x_fake.view(-1, 1, 28, 28), nrow=8, normalize=True
    )

    # ---- Plot ----
    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 1)
    plt.imshow(real_grid.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.title("Real Images")

    plt.subplot(2, 1, 2)
    plt.imshow(fake_grid.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.title("Generated Samples")

    plt.tight_layout()
    plt.show()

def show_reconstructions(model, data_loader, device):
    model.eval()
    x, _ = next(iter(data_loader))
    x = x.to(device)[:8]

    with torch.no_grad():
        z = model(x)
        x_recon = model.inverse(z)

    x = x.cpu()
    x_recon = x_recon.cpu()

    comparison = torch.cat([x, x_recon])
    grid = vutils.make_grid(comparison, nrow=8, normalize=True)

    plt.figure(figsize=(8,4))
    plt.imshow(grid.permute(1,2,0))
    plt.axis('off')
    plt.title("Top: Original | Bottom: Reconstruction")
    plt.show()

import numpy as np

def plot_latent_distribution(model, data_loader, device):
    model.eval()
    zs = []

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            z = model(x)
            zs.append(z.cpu().numpy())

    zs = np.concatenate(zs, axis=0)

    plt.figure()
    plt.hist(zs.flatten(), bins=100, density=True)
    plt.title("Latent Distribution")
    plt.xlabel("z values")
    plt.ylabel("Density")
    plt.show()

def interpolate(model, data_loader, device, steps=10):
    model.eval()
    x, _ = next(iter(data_loader))
    x1, x2 = x[0:1].to(device), x[1:2].to(device)

    with torch.no_grad():
        z1 = model(x1)
        z2 = model(x2)

    interpolations = []
    for alpha in torch.linspace(0,1,steps):
        z = (1-alpha)*z1 + alpha*z2
        x_interp = model.inverse(z)
        interpolations.append(x_interp.cpu())

    interpolations = torch.cat(interpolations)
    grid = vutils.make_grid(interpolations, nrow=steps, normalize=True)

    plt.figure(figsize=(10,2))
    plt.imshow(grid.permute(1,2,0))
    plt.axis('off')
    plt.title("Latent Interpolation")
    plt.show()

if __name == 
