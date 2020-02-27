from collections import OrderedDict

import torch
import torch.nn as nn
from torch.distributions.multinomial import Multinomial


class SimpleModel(nn.Module):
    """SimpleModel represents a lightweight model for checking codes.

    This model is quite simple to check codes quickly.

    Attributes
    ----------
    self.num_classes int : number of classes of dataset.
    self.layers nn.ModuleDict : ModuleDict of models.
    """
    def __init__(self, num_classes):
        """
        Parameters
        ----------
        num_classes int : number of classes of dataset.
        """
        super(SimpleModel, self).__init__()
        self.num_classes = num_classes
        self.layers = nn.ModuleList([
            nn.Conv2d(3, 9, 3, padding=1, stride=1),
            nn.GroupNorm(3, 9),
            nn.ELU(),
            nn.Conv2d(9, self.num_classes, 1, padding=0, stride=2),
            nn.AdaptiveAvgPool2d(1)
        ])

    def forward(self, x):
        """
        Parameters
        ----------
        x torch.Tensor : input tensor whose shape is [b, c, h, w].

        Returns
        -------
        torch.squeeze(x) torch.Tensor : logit tensor which will be input of softmax.
        """
        for layer in self.layers:
            x = layer(x)
        return torch.reshape(x, x.shape[:2])  # [b, num_classes]


class Model(nn.Module):
    """Model represents a model mainly used in experiments.

    Attributes
    ----------
    self.num_classes int : number of classes of dataset.
    self.layers nn.ModuleDict : ModuleDict of models.
    """
    def __init__(self, num_classes):
        """
        Parameters
        ----------
        num_classes int : number of classes of dataset.
        """
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.layers = nn.ModuleDict(OrderedDict([
            # CONV-GN-ELU
            ("conv1", nn.Conv2d(3, 96, 3, padding=1, stride=1)),
            ("GN1", nn.GroupNorm(3, 96)),
            ("ELU1", nn.ELU()),
            # CONV-GN-ELU * 2 + Dropout
            ("conv2", nn.Conv2d(96, 96, 3, padding=1, stride=1)),
            ("GN2", nn.GroupNorm(3, 96)),
            ("ELU2", nn.ELU()),
            ("conv3", nn.Conv2d(96, 96, 3, padding=1, stride=2)),
            ("GN3", nn.GroupNorm(3, 96)),
            ("ELU3", nn.ELU()),
            ("DO1", nn.Dropout(0.5)),
            # CONV-GN-ELU * 3 + Dropout
            ("conv4", nn.Conv2d(96, 192, 3, padding=1, stride=1)),
            ("GN4", nn.GroupNorm(6, 192)),
            ("ELU4", nn.ELU()),
            ("conv5", nn.Conv2d(192, 192, 3, padding=1, stride=1)),
            ("GN5", nn.GroupNorm(6, 192)),
            ("ELU5", nn.ELU()),
            ("conv6", nn.Conv2d(192, 192, 3, padding=1, stride=2)),
            ("GN6", nn.GroupNorm(6, 192)),
            ("ELU6", nn.ELU()),
            ("DO2", nn.Dropout(0.5)),
            # CONV-GN-ELU * 2 + CONV + GAP
            ("conv7", nn.Conv2d(192, 192, 3, padding=1, stride=1)),
            ("GN7", nn.GroupNorm(6, 192)),
            ("ELU7", nn.ELU()),
            ("conv8", nn.Conv2d(192, 192, 1, padding=0, stride=1)),
            ("GN8", nn.GroupNorm(6, 192)),
            ("ELU8", nn.ELU()),
            ("conv9", nn.Conv2d(192, self.num_classes, 1, padding=0, stride=2)),
            ("pool", nn.AdaptiveAvgPool2d(1))
        ]))

    def forward(self, x):
        """
        Parameters
        ----------
        x torch.Tensor : input tensor whose shape is [b, c, h, w].

        Returns
        -------
        torch.squeeze(x) torch.Tensor : logit tensor which will be input of softmax.
        """
        for layer in self.layers.values():
            x = layer(x)
        return torch.reshape(x, x.shape[:2])  # [b, num_classes]


class StochasticActivationPruning(nn.Module):
    """SimpleModel represents a nn.Module of Stochastic Activation Pruning.

    The original paper is https://arxiv.org/abs/1803.01442.

    Attributes
    ----------
    self.ratio float : ratio of pruning which can be larger than 1.0.
    self.is_valid bool : if this flag is True, inject SAP.
    """
    def __init__(self, ratio=1.0, is_valid=False):
        """
        Parameters
        ----------
        ratio float : ratio of pruning which can be larger than 1.0.
        is_valid bool : if this flag is True, inject SAP.
        """
        super(StochasticActivationPruning, self).__init__()
        self.ratio = ratio
        self.is_valid = is_valid

    def forward(self, inputs):
        """

        If self.training or not self.is_valid, just return inputs.
        If self.is_valid apply SAP to inputs and return the result tensor.

        Parameters
        ----------
        inputs torch.Tensor : input tensor whose shape is [b, c, h, w].

        Returns
        -------
        outputs torch.Tensor : just return inputs or stochastically pruned inputs.
        """
        if self.training or not self.is_valid:
            return inputs
        else:
            b, c, h, w = inputs.shape
            inputs_1d = inputs.reshape([b, c * h * w])  # [b, c * h * w]
            outputs = torch.zeros_like(inputs_1d)  # outputs with 0 initilization
            inputs_1d_sum = torch.sum(torch.abs(inputs_1d), dim=-1, keepdim=True)
            inputs_1d_prob = torch.abs(inputs_1d) / inputs_1d_sum

            repeat_num = int(c * h * w * self.ratio)
            idx = Multinomial(repeat_num, inputs_1d_prob).sample()
            outputs[idx.nonzero(as_tuple=True)] = inputs_1d[idx.nonzero(as_tuple=True)]
            outputs = outputs / (1 - (1 - inputs_1d_prob) ** repeat_num + 1e-12)
            outputs = outputs.reshape([b, c, h, w])  # [b, c, h, w]

        return outputs


class ModelSAP(nn.Module):
    """Model represents a model mainly used in experiments.

    Attributes
    ----------
    self.num_classes int : number of classes of dataset.
    self.layers nn.ModuleDict : ModuleDict of models.
    """
    def __init__(self, num_classes):
        """
        Parameters
        ----------
        num_classes int : number of classes of dataset.
        """
        super(ModelSAP, self).__init__()
        self.num_classes = num_classes
        self.layers = nn.ModuleDict(OrderedDict([
            # CONV-GN-ELU
            ("conv1", nn.Conv2d(3, 96, 3, padding=1, stride=1)),
            ("GN1", nn.GroupNorm(3, 96)),
            ("ELU1", nn.ELU()),
            ("SAP1", StochasticActivationPruning()),
            # CONV-GN-ELU * 2 + Dropout
            ("conv2", nn.Conv2d(96, 96, 3, padding=1, stride=1)),
            ("GN2", nn.GroupNorm(3, 96)),
            ("ELU2", nn.ELU()),
            ("SAP2", StochasticActivationPruning()),
            ("conv3", nn.Conv2d(96, 96, 3, padding=1, stride=2)),
            ("GN3", nn.GroupNorm(3, 96)),
            ("ELU3", nn.ELU()),
            ("SAP3", StochasticActivationPruning()),
            ("DO1", nn.Dropout(0.5)),
            # CONV-GN-ELU * 3 + Dropout
            ("conv4", nn.Conv2d(96, 192, 3, padding=1, stride=1)),
            ("GN4", nn.GroupNorm(6, 192)),
            ("ELU4", nn.ELU()),
            ("SAP4", StochasticActivationPruning()),
            ("conv5", nn.Conv2d(192, 192, 3, padding=1, stride=1)),
            ("GN5", nn.GroupNorm(6, 192)),
            ("ELU5", nn.ELU()),
            ("SAP5", StochasticActivationPruning()),
            ("conv6", nn.Conv2d(192, 192, 3, padding=1, stride=2)),
            ("GN6", nn.GroupNorm(6, 192)),
            ("ELU6", nn.ELU()),
            ("SAP6", StochasticActivationPruning()),
            ("DO2", nn.Dropout(0.5)),
            # CONV-GN-ELU * 2 + CONV + GAP
            ("conv7", nn.Conv2d(192, 192, 3, padding=1, stride=1)),
            ("GN7", nn.GroupNorm(6, 192)),
            ("ELU7", nn.ELU()),
            ("SAP7", StochasticActivationPruning(is_valid=True)),
            ("conv8", nn.Conv2d(192, 192, 1, padding=0, stride=1)),
            ("GN8", nn.GroupNorm(6, 192)),
            ("ELU8", nn.ELU()),
            ("SAP8", StochasticActivationPruning(is_valid=True)),
            ("conv9", nn.Conv2d(192, self.num_classes, 1, padding=0, stride=2)),
            ("pool", nn.AdaptiveAvgPool2d(1))
        ]))

    def forward(self, x):
        """
        Parameters
        ----------
        x torch.Tensor : input tensor whose shape is [b, c, h, w].

        Returns
        -------
        torch.squeeze(x) torch.Tensor : logit tensor which will be input of softmax.
        """
        for layer in self.layers.values():
            x = layer(x)
        return torch.reshape(x, x.shape[:2])  # [b, num_classes]
