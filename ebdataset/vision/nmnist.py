import os
import numpy as np
from torch.utils import data
from .parsers.aer import readAERFile


class NMnist(data.Dataset):
    """
    NMnist dataset from
    Orchard, G.; Cohen, G.; Jayawant, A.; and Thakor, N.
    “Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades",
    Frontiers in Neuroscience, vol.9, no.437, Oct. 2015

    Available for download: https://www.garrickorchard.com/datasets/n-mnist
    """

    def __init__(self, path: str, is_train: bool = True, transforms=None):
        path = os.path.join(path, "Train" if is_train else "Test")
        assert os.path.exists(path)
        self._files = []
        self._labels = []

        for root, dirs, files in os.walk(path):
            digit = os.path.basename(root)
            for file in files:
                if file.endswith(".bin"):
                    self._files.append(os.path.join(root, file))
                    self._labels.append(int(digit))

        self._files = np.asarray(self._files)
        self._labels = np.asarray(self._labels)
        self.transforms = transforms

    def __len__(self):
        return self._files.size

    def __getitem__(self, index):
        spike_train = readAERFile(self._files[index])
        spike_train.width = 34
        spike_train.height = 34
        spike_train.duration = spike_train.ts.max() + 1
        if self.transforms is not None:
            spike_train = self.transforms(spike_train)
        return spike_train, self._labels[index]
