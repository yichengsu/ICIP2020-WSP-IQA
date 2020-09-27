import warnings

import torch
import torch.nn as nn


class HuberLoss(nn.Module):
    r"""
    Args:
        delta: Default: ``1``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
    """
    __constants__ = ['reduction']

    def __init__(self, delta=1, reduction='mean'):
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, input, target):
        if not (target.size() == input.size()):
            warnings.warn("Using a target size ({}) that is different to the input size ({}). "
                          "This will likely lead to incorrect results due to broadcasting. "
                          "Please ensure they have the same size.".format(
                              target.size(), input.size()), stacklevel=2)

        # https://github.com/pytorch/pytorch/blob/a460c856dd7dafa706a91a985513571fe1a4e3be/torch/csrc/api/include/torch/nn/functional/loss.h#L231-L267
        t = torch.abs(input - target)
        ret = torch.where(t <= self.delta, 0.5*torch.pow(t, 2), self.delta*(t-self.delta/2))
        ret = torch.sum(ret, dim=1)
        if self.reduction:
            ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)
        return ret
