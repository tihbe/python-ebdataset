import os
import numpy as np
from torch.utils import data
from .parsers.aedat import readAEDATv2_davies


class INIUCF50(data.Dataset):
    """INI UCF-50 dataset from:
    Hu, Y., Liu, H., Pfeiffer, M., and Delbruck, T. (2016).
    DVS Benchmark Datasets for Object Tracking, Action Recognition and Object Recognition.
    Front. Neurosci. 10, 405. doi:10.3389/fnins.2016.00405.

    Available for download: https://dgyblog.com/projects-term/dvs-dataset.html
    """

    def __init__(self, path: str, transforms=None):
        assert os.path.exists(path)
        self._files = []
        self._labels = []

        for root, dirs, files in os.walk(path):
            label = os.path.basename(root)
            for file in files:
                if file.endswith(".aedat"):
                    self._files.append(os.path.join(root, file))
                    self._labels.append(label)
        self._files = np.asarray(self._files)
        self._labels = np.asarray(self._labels)
        self.transforms = transforms

    def __len__(self):
        return self._labels.size

    def __getitem__(self, index):
        spike_train = readAEDATv2_davies(self._files[index])
        spike_train.width = 240
        spike_train.height = 180
        spike_train.duration = spike_train.ts.max() + 1
        if self.transforms is not None:
            spike_train = self.transforms(spike_train)
        return spike_train, self._labels[index]
