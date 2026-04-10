import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. WASSERSTEIN (SLICED)
# =========================================================

def wasserstein_1d(x, y):
    x_sorted, _ = torch.sort(x)
    y_sorted, _ = torch.sort(y)
    return torch.mean(torch.abs(x_sorted - y_sorted))


def sliced_wasserstein(z, n_projections=100):
    """
    z: (batch_size, dim)
    """
    device = z.device
    dim = z.size(1)

    z_ref = torch.randn_like(z)
    swd = 0.0

    for _ in range(n_projections):
        direction = torch.randn(dim, device=device)
        direction = F.normalize(direction, dim=0)

        proj_z = z @ direction
        proj_ref = z_ref @ direction

        swd += wasserstein_1d(proj_z, proj_ref)

    return swd / n_projections


# =========================================================
# 2. MMD (Maximum Mean Discrepancy)
# =========================================================

def gaussian_kernel(x, y, sigma=1.0):
    x = x.unsqueeze(1)  # (N,1,D)
    y = y.unsqueeze(0)  # (1,M,D)
    return torch.exp(-((x - y) ** 2).sum(2) / (2 * sigma ** 2))


def compute_mmd(x, y, sigma=1.0):
    """
    x, y: (batch_size, dim)
    """
    Kxx = gaussian_kernel(x, x, sigma)
    Kyy = gaussian_kernel(y, y, sigma)
    Kxy = gaussian_kernel(x, y, sigma)

    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()


def mmd_to_gaussian(z, sigma=1.0):
    z_ref = torch.randn_like(z)
    return compute_mmd(z, z_ref, sigma)


# =========================================================
# 3. EXTRACT INTERMEDIATE REPRESENTATIONS
# =========================================================
def forward_with_intermediates(model, x, model_type):
    if model_type == "nice":
        return forward_with_intermediates_nice(model, x)
    elif model_type == "realnvp":
        return forward_with_intermediates_realnvp(model, x)
    elif model_type == "glow":
        return forward_with_intermediates_glow(model, x)

def forward_with_intermediates_nice(model, x):
    """
    Assumes model.layers exists (adapt if needed)
    """
    zs = []
    z = x

    for layer in model.layers:
        z = layer(z)
        zs.append(z)
    z = model.scaling_layer(z)
    zs.append(z)

    return zs


# =========================================================
# 4. COMPUTE DISTANCE ACROSS LAYERS
# =========================================================

def compute_distances_across_layers(
    model,
    data_loader,
    device,
    metric="wasserstein",
    n_projections=100,
    model_type = "nice"
):
    model.eval()
    layer_distances = []

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)

            zs = forward_with_intermediates(model, x, model_type)

            for i, z in enumerate(zs):

                if metric == "wasserstein":
                    d = sliced_wasserstein(z, n_projections)
                elif metric == "mmd":
                    d = mmd_to_gaussian(z)
                else:
                    raise ValueError("Unknown metric")

                if len(layer_distances) <= i:
                    layer_distances.append([])

                layer_distances[i].append(d.item())

    # Average over batches
    layer_distances = [np.mean(layer) for layer in layer_distances]

    return layer_distances


# =========================================================
# 5. TRACK DURING TRAINING (EPOCH-WISE)
# =========================================================

def track_during_training(
    model,
    data_loader,
    device,
    selected_layers=[0, 2, 4],
    metric="wasserstein",
    n_projections=20
):
    """
    Returns dict:
    {layer_index: [distance over epochs]}
    """
    model.eval()
    results = {l: [] for l in selected_layers}

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            zs = forward_with_intermediates(model, x)

            for l in selected_layers:
                z = zs[l]

                if metric == "wasserstein":
                    d = sliced_wasserstein(z, n_projections)
                else:
                    d = mmd_to_gaussian(z)

                results[l].append(d.item())

            break  # use only one batch for speed

    return results


# =========================================================
# 6. PLOTTING FUNCTIONS
# =========================================================

def plot_layer_vs_distance(distances_dict, title="Layer vs Distance to Gaussian"):
    """
    distances_dict:
    {
        "NICE": [...],
        "RealNVP": [...],
        "GLOW": [...]
    }
    """
    plt.figure()

    for name, distances in distances_dict.items():
        plt.plot(distances, marker='o', label=name)

    plt.xlabel("Layer")
    plt.ylabel("Distance to Gaussian")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def plot_epoch_vs_distance(epoch_results, title="Epoch vs Distance"):
    """
    epoch_results:
    {
        layer_idx: [values over epochs]
    }
    """
    plt.figure()

    for layer, values in epoch_results.items():
        plt.plot(values, marker='o', label=f"Layer {layer}")

    plt.xlabel("Epoch checkpoint")
    plt.ylabel("Distance to Gaussian")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

from sklearn.decomposition import PCA


def plot_pca_evolution(model, data_loader, device, layers_to_plot=[0, 2, 4]):
    model.eval()

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            zs = forward_with_intermediates(model, x)
            break  # one batch is enough

    for l in layers_to_plot:
        z = zs[l].cpu().numpy()

        # Gaussian reference
        z_ref = np.random.randn(*z.shape)

        # Combine
        combined = np.concatenate([z, z_ref], axis=0)

        # PCA
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(combined)

        z_pca = reduced[:len(z)]
        ref_pca = reduced[len(z):]

        # Plot
        plt.figure()
        plt.scatter(z_pca[:, 0], z_pca[:, 1], alpha=0.5, label='Model')
        plt.scatter(ref_pca[:, 0], ref_pca[:, 1], alpha=0.5, label='Gaussian')
        plt.title(f"PCA at Layer {l}")
        plt.legend()
        plt.grid()
        plt.show()