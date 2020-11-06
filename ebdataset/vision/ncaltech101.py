import os
import numpy as np
from torch.utils import data
from .parsers.aer import readAERFile


class NCaltech101(data.Dataset):
    """
    NCaltech101 dataset from
    Orchard, G.; Cohen, G.; Jayawant, A.; and Thakor, N.
    “Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades",
    Frontiers in Neuroscience, vol.9, no.437, Oct. 2015

    Available for download: https://www.garrickorchard.com/datasets/n-caltech101
    """

    def __init__(self, path: str, transforms=None):
        assert os.path.exists(path)
        self._files = []
        self._labels = []
        self.transforms = transforms
        for root, dirs, files in os.walk(path):
            label = os.path.basename(root)
            for file in files:
                if file.endswith(".bin"):
                    self._files.append(os.path.join(root, file))
                    self._labels.append(label)

        self._files = np.array(self._files)
        self._labels = np.array(self._labels)

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        spike_train = readAERFile(self._files[index])
        spike_train.width = 34
        spike_train.height = 34
        spike_train.duration = spike_train.ts.max() + 1
        if self.transforms is not None:
            spike_train = self.transforms(spike_train)
        return spike_train, self._labels[index]
