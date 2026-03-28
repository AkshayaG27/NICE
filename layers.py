# layers.py

import torch
import torch.nn as nn

pick_even_columns = lambda xs: xs[:, 0::2]
pick_odd_columns  = lambda xs: xs[:, 1::2]


def reweave_columns(first, second, which_was_first):
    cols = []
    if which_was_first == 'even':
        for k in range(second.shape[1]):
            cols.append(first[:, k])
            cols.append(second[:, k])
        if first.shape[1] > second.shape[1]:
            cols.append(first[:, -1])
    else:
        for k in range(first.shape[1]):
            cols.append(second[:, k])
            cols.append(first[:, k])
        if second.shape[1] > first.shape[1]:
            cols.append(second[:, -1])
    return torch.stack(cols, dim=1)


class HalfAndHalfLayer(nn.Module):
    def __init__(self, dim, which_half, small_net):
        super().__init__()
        self.dim = dim
        assert which_half in ['even', 'odd'], \
            "which_half must be 'even' or 'odd'"
        self.which_half = which_half
        if which_half == 'even':
            self.frozen_half   = pick_even_columns
            self.changing_half = pick_odd_columns
        else:
            self.frozen_half   = pick_odd_columns
            self.changing_half = pick_even_columns
        self.add_module('small_net', small_net)

    def forward(self, x):
        frozen      = self.frozen_half(x)
        changing    = self.changing_half(x)
        transformed = self.transform_second_half(changing, self.small_net(frozen))
        return reweave_columns(frozen, transformed, self.which_half)

    def inverse(self, y):
        frozen        = self.frozen_half(y)
        changing      = self.changing_half(y)
        untransformed = self.untransform_second_half(changing, self.small_net(frozen))
        return reweave_columns(frozen, untransformed, self.which_half)

    def transform_second_half(self, changing, net_output):
        raise NotImplementedError

    def untransform_second_half(self, changing, net_output):
        raise NotImplementedError


class ShiftCouplingLayer(HalfAndHalfLayer):
    """Forward: changing + net(frozen). Inverse: changing - net(frozen)."""
    def transform_second_half(self, changing, net_output):
        return changing + net_output

    def untransform_second_half(self, changing, net_output):
        return changing - net_output


class ScaleCouplingLayer(HalfAndHalfLayer):
    """Forward: changing * net(frozen). Inverse: changing / net(frozen)."""
    def transform_second_half(self, changing, net_output):
        return torch.mul(changing, net_output)

    def untransform_second_half(self, changing, net_output):
        return torch.mul(changing, torch.reciprocal(net_output))


class ShiftAndScaleCouplingLayer(HalfAndHalfLayer):
    """Forward: changing * scale + shift. Inverse: not yet implemented."""
    def transform_second_half(self, changing, net_output):
        scale = self.frozen_half(net_output)
        shift = self.changing_half(net_output)
        return torch.mul(changing, scale) + shift

    def untransform_second_half(self, changing, net_output):
        raise NotImplementedError(
            "ShiftAndScaleCouplingLayer inverse not yet implemented."
        )