import os
import numpy as np
from h5py import File
from torch.utils import data


class NTidigits(data.Dataset):
    """NTidigits dataset from:
    Anumula, Jithendar, et al. “Feature Representations for Neuromorphic Audio Spike Streams.”
    Frontiers in Neuroscience, vol. 12, Feb. 2018, p. 23. DOI.org (Crossref), doi:10.3389/fnins.2018.00023.

    Available for download at https://docs.google.com/document/d/1Uxe7GsKKXcy6SlDUX4hoJVAC0-UkH-8kr5UXp0Ndi1M
    """

    def __init__(self, path: str, is_train=True, transforms=None, only_single_digits=False):
        assert os.path.exists(path)
        self.prename = "train" if is_train else "test"
        self.path = path
        self.transforms = transforms

        with File(path, "r") as f:
            self.samples = f[self.prename + "_labels"][()]

        if only_single_digits:
            self.samples = list(filter(lambda s: len(NTidigits._get_label_for_sample(s)) == 1, self.samples))

    @staticmethod
    def _get_label_for_sample(sample_id):
        return sample_id.decode("utf-8").split("-")[-1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_id = self.samples[index]
        with File(self.path, "r") as f:
            addresses = f[self.prename + "_addresses"][sample_id][()]
            ts = f[self.prename + "_timestamps"][sample_id][()]

        sparse_spike_train = np.recarray(shape=len(ts), dtype=[("addr", addresses.dtype), ("ts", ts.dtype)])
        sparse_spike_train.addr = addresses
        sparse_spike_train.ts = ts

        if self.transforms is not None:
            sparse_spike_train = self.transforms(sparse_spike_train)

        return sparse_spike_train, NTidigits._get_label_for_sample(sample_id)
