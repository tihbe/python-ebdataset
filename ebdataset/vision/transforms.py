"""
Torchvision-like transforms for 2d sparse rec-array spike trains
"""
import torch
import numpy as np
import quantities as units
from torchvision.transforms import Compose


class ScaleDown(object):
    """Scale down a 2d sparse spike train by factor (both in x and y)"""

    def __init__(self, width, height, factor):
        self.authorized_x = list(range(0, width, factor))
        self.authorized_y = list(range(0, height, factor))
        self.factor = factor

    def __call__(self, sparse_spike_train):
        x_mask = np.isin(sparse_spike_train.x, self.authorized_x)
        y_mask = np.isin(sparse_spike_train.y, self.authorized_y)
        mask = x_mask & y_mask

        out = np.recarray(np.sum(mask), dtype=sparse_spike_train.dtype)
        out.x = sparse_spike_train.x[mask] // self.factor
        out.y = sparse_spike_train.y[mask] // self.factor
        out.p = sparse_spike_train.p[mask]
        out.ts = sparse_spike_train.ts[mask]

        return out


class MaxTime(object):
    """Limit the time of a 2d sparse spike train"""

    def __init__(self, max_time: units.UnitTime, dt: units.UnitTime = 1 * units.us):
        self.max = (max_time.rescale(dt.units) / dt).magnitude

    def __call__(self, sparse_spike_train):
        mask = sparse_spike_train.ts < self.max
        out = np.recarray(np.sum(mask), dtype=sparse_spike_train.dtype)
        out.x = sparse_spike_train.x[mask]
        out.y = sparse_spike_train.y[mask]
        out.p = sparse_spike_train.p[mask]
        out.ts = sparse_spike_train.ts[mask]

        return out


class ToDense(object):
    """Transform a sparse spike train to a dense torch tensor of shape (x, y, p, time)
    with time unit defined by dt. Time accumulation is done with a max function."""

    def __init__(
        self,
        dt: units.UnitTime,  # Time scale of dense tensor
    ):
        self.dt = dt

    def __call__(self, sparse_spike_train):
        time_scale = ((sparse_spike_train.time_scale * units.second).rescale(self.dt.units) / self.dt).magnitude
        duration = np.ceil(sparse_spike_train.duration * time_scale).astype(int)
        dense_spike_train = torch.zeros((sparse_spike_train.width, sparse_spike_train.height, 2, duration))

        ts = (sparse_spike_train.ts * time_scale).astype(int)

        dense_spike_train[
            sparse_spike_train.x.astype(int), sparse_spike_train.y.astype(int), sparse_spike_train.p.astype(int), ts
        ] = 1

        return dense_spike_train


class Flatten(object):
    """Flatten a dense 2d spike train (x, y, p) to 1d over time"""

    def __call__(self, dense_spike_train):
        return dense_spike_train.reshape(-1, dense_spike_train.shape[-1])
