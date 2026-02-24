"""CTGAN module."""

import warnings

import numpy as np
import pandas as pd
import torch
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, Parameter, ReLU, Sequential, functional, Sigmoid

class Discriminator(Module):
    """Discriminator for the CTGAN."""

    def __init__(self, input_dim, discriminator_dim, pac=10, loss='wasserstein', 
                 bayes=False, prior_dist='gauss', std_act='exp', prior_mu=0, prior_sigma=0.1):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            linearlayer = BayesLinear(prior_dist = prior_dist, std_act = std_act, prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=dim, out_features=item) if bayes else Linear(dim, item)
            seq += [linearlayer, LeakyReLU(0.2), Dropout(0.5)]
            dim = item
        seq += [Linear(dim, 1)]
        if loss=='vanilla': seq += [Sigmoid()]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))


class Residual(Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o, bayes=False, 
                 prior_dist='gauss', std_act='exp', prior_mu=0, prior_sigma=0.1):
        super(Residual, self).__init__()
        self.fc = BayesLinear(prior_dist = prior_dist, std_act = std_act, prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=i, out_features=o) if bayes else Linear(i, o) 
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(Module):
    """Generator for the CTGAN."""

    def __init__(self, embedding_dim, generator_dim, data_dim, bayes=False, 
                 prior_dist='gauss', std_act='exp', prior_mu=0, prior_sigma=0.1):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item, bayes, prior_dist, std_act, prior_mu, prior_sigma)]
            dim += item
        linear_layer = BayesLinear(prior_dist = prior_dist, std_act = std_act, prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=dim, out_features=data_dim) if bayes else Linear(dim, data_dim)
        seq.append(linear_layer)
        self.seq = Sequential(*seq)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data


def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]
