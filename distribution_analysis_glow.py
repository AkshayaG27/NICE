# ------------------------------------------------------------------------------------------------------------------------------------------
# Model_type is GLOW
# ------------------------------------------------------------------------------------------------------------------------------------------

import re
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from models import Glow
from utils import get_dataloader

def preprocess_for_glow(x, n_bits=5):
    n_bins = 2.0 ** n_bits

    x = x * 255

    if n_bits < 8:
        x = torch.floor(x / 2 ** (8 - n_bits))

    x = x / n_bins - 0.5

    x = x + torch.rand_like(x) / n_bins

    return x



# =========================================================
# 0. LOG PARSING
# =========================================================

def parse_losses_from_log(file_path):
    train_losses = []
    val_losses = []

    with open(file_path, "r") as f:
        for line in f:
            if "train:" in line and "val:" in line:
                train_match = re.search(r"train:\s*([-+]?\d*\.?\d+)", line)
                val_match   = re.search(r"val:\s*([-+]?\d*\.?\d+)", line)

                if train_match and val_match:
                    train_losses.append(float(train_match.group(1)))
                    val_losses.append(float(val_match.group(1)))

    return train_losses, val_losses


# =========================================================
# 1. WASSERSTEIN (SLICED)
# =========================================================

def wasserstein_1d(x, y):
    x_sorted, _ = torch.sort(x)
    y_sorted, _ = torch.sort(y)
    return torch.mean(torch.abs(x_sorted - y_sorted))


def sliced_wasserstein(z, n_projections=100):
    z = z.reshape(z.size(0), -1)
    dim = z.size(1)

    z_ref = torch.randn_like(z)

    swd = 0.0
    for _ in range(n_projections):
        direction = torch.randn(dim, device=z.device)
        direction = F.normalize(direction, dim=0)

        proj_z   = z @ direction
        proj_ref = z_ref @ direction

        swd += wasserstein_1d(proj_z, proj_ref)

    return swd / n_projections


# =========================================================
# 2. MMD
# =========================================================

def gaussian_kernel(x, y, sigma=1.0):
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    return torch.exp(-((x - y) ** 2).sum(2) / (2 * sigma ** 2))


def compute_mmd(x, y, sigma=1.0):
    Kxx = gaussian_kernel(x, x, sigma)
    Kyy = gaussian_kernel(y, y, sigma)
    Kxy = gaussian_kernel(x, y, sigma)

    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()


def mmd_to_gaussian(z, sigma=1.0):
    z = z.reshape(z.size(0), -1)
    z_ref = torch.randn_like(z)
    return compute_mmd(z, z_ref, sigma)


# =========================================================
# 3. GLOW LATENT EVOLUTION EXTRACTOR
# =========================================================

def glow_forward_with_intermediates(model, x):
    """
    Returns cumulative latent representations after each Glow block.

    This is the scientifically correct notion of latent evolution for Glow:
    after each block we concatenate all latents emitted so far.
    """
    model.eval()

    cumulative_z = []
    intermediates = []

    with torch.no_grad():
        out = x

        for block_idx, block in enumerate(model.blocks):

            # -------------------------
            # SQUEEZE OPERATION
            # -------------------------
            b, c, h, w = out.shape

            squeezed = out.view(b, c, h // 2, 2, w // 2, 2)
            squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
            out = squeezed.contiguous().view(
                b, c * 4, h // 2, w // 2
            )

            # -------------------------
            # FLOW STEPS
            # -------------------------
            for flow in block.flows:
                out, _ = flow(out)

            # -------------------------
            # SPLIT LATENT
            # -------------------------
            if block.split:
                out, z_new = out.chunk(2, 1)
                cumulative_z.append(z_new.flatten(1))

            else:
                cumulative_z.append(out.flatten(1))

            full_latent = torch.cat(cumulative_z, dim=1)
            intermediates.append(full_latent.detach())

    return intermediates


# =========================================================
# 4. DISTANCE ACROSS GLOW BLOCKS
# =========================================================

def compute_distances_across_layers(
    model,
    data_loader,
    device,
    metric="wasserstein",
    n_projections=100,
):
    model.eval()

    layer_distances = []

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            x = preprocess_for_glow(x)
            zs = glow_forward_with_intermediates(model, x)

            for i, z in enumerate(zs):

                if metric == "wasserstein":
                    d = sliced_wasserstein(z, n_projections)

                elif metric == "mmd":
                    d = mmd_to_gaussian(z)

                else:
                    raise ValueError("Metric must be 'wasserstein' or 'mmd'")

                if len(layer_distances) <= i:
                    layer_distances.append([])

                layer_distances[i].append(d.item())

    return [np.mean(layer) for layer in layer_distances]


# =========================================================
# 5. EPOCH-WISE TRACKING
# =========================================================

def track_during_training(
    model,
    data_loader,
    device,
    selected_layers=None,
    metric="wasserstein",
    n_projections=20,
):
    model.eval()

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            x = preprocess_for_glow(x)
            zs = glow_forward_with_intermediates(model, x)
            break

    if selected_layers is None:
        selected_layers = list(range(len(zs)))

    results = {}

    for l in selected_layers:
        if l >= len(zs):
            continue

        z = zs[l]

        if metric == "wasserstein":
            d = sliced_wasserstein(z, n_projections)
        else:
            d = mmd_to_gaussian(z)

        results[l] = d.item()

    return results


# =========================================================
# 6. PLOTTING
# =========================================================

def plot_layer_vs_distance(distances, title="Glow Block vs Distance to Gaussian"):
    plt.figure()

    plt.plot(distances, marker='o')

    plt.xlabel("Glow Block")
    plt.ylabel("Distance to Gaussian")
    plt.title(title)

    plt.grid()
    plt.tight_layout()
    plt.savefig("layer_vs_distance.png")
    plt.show()


def plot_epoch_vs_distance(epoch_results, title="Epoch vs Distance"):
    plt.figure()

    for layer, values in epoch_results.items():
        plt.plot(values, marker='o', label=f"Layer {layer}")

    plt.xlabel("Epoch Checkpoint")
    plt.ylabel("Distance to Gaussian")
    plt.title(title)

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("epoch_vs_distance.png")
    plt.show()


def plot_pca_evolution(model, data_loader, device, layers_to_plot=None):
    model.eval()

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            x = preprocess_for_glow(x)
            zs = glow_forward_with_intermediates(model, x)
            break

    if layers_to_plot is None:
        layers_to_plot = list(range(len(zs)))

    for l in layers_to_plot:

        if l >= len(zs):
            print(f"Layer {l} unavailable. Skipping.")
            continue

        z = zs[l].cpu().numpy()

        z_ref = np.random.randn(*z.shape)

        combined = np.concatenate([z, z_ref], axis=0)

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(combined)

        z_pca   = reduced[:len(z)]
        ref_pca = reduced[len(z):]

        plt.figure()

        plt.scatter(z_pca[:, 0], z_pca[:, 1], alpha=0.5, label="Glow Latent")
        plt.scatter(ref_pca[:, 0], ref_pca[:, 1], alpha=0.5, label="Gaussian")

        plt.title(f"PCA at Glow Block {l}")

        plt.legend()
        plt.grid()
        plt.tight_layout()

        plt.savefig(f"pca_block_{l}.png")
        plt.show()



# =========================================================
# 7. MAIN
# =========================================================

if __name__ == "__main__":

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 512
    CHECKPOINT_PATH = "checkpoints_GLOW/model_008001.pt"

    METRIC = "wasserstein"

    print("Loading test data...")
    test_loader = get_dataloader(
        "mnist",
        BATCH_SIZE,
        split="test",
        flatten_input=False
    )

    print("Loading Glow model...")
    model = Glow(
        in_channel=1,
        n_flow=24,
        n_block=3,
        affine=True,
        conv_lu=True
    ).to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    if all(k.startswith("module.") for k in checkpoint.keys()):
        checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}

    model.load_state_dict(checkpoint)
    model.eval()

    print(f"Computing {METRIC} distances...")
    distances = compute_distances_across_layers(
        model=model,
        data_loader=test_loader,
        device=DEVICE,
        metric=METRIC,
        n_projections=100,
    )

    print("\nGlow Latent Gaussianity Across Blocks:")
    for i, d in enumerate(distances):
        print(f"  Block {i}: {d:.6f}")

    plot_layer_vs_distance(distances)

    print("\nRunning PCA evolution...")
    plot_pca_evolution(
        model=model,
        data_loader=test_loader,
        device=DEVICE,
        layers_to_plot=[0, 1, 2],
    )

    print("\nAnalysis complete.")