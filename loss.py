# loss.py

import torch
import torch.nn.functional as F


class StandardNormal:
    """
    Gaussian prior — good for MNIST and TFD.
    log p(z) = -0.5 * sum(z^2)
    """
    def log_prob(self, z):
        return -0.5 * z.pow(2).sum(dim=1)


class StandardLogistic:
    """
    Logistic prior — better for CIFAR-10 and SVHN.
    log p(z) = -sum( softplus(z) + softplus(-z) )
    F.softplus is numerically stable; torch.log(1 + exp(z)) is not.
    """
    def log_prob(self, z):
        return -(F.softplus(z) + F.softplus(-z)).sum(dim=1)


def nll_loss(model, x, prior):
    """
    Negative log-likelihood = -mean( log p(z) + log|det J| )
    This is the single training objective for all datasets.
    """
    z      = model.encode(x)
    log_pz = prior.log_prob(z)
    ldj    = model.log_det_jacobian()
    return -(log_pz + ldj).mean()