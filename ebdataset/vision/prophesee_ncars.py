import os
import numpy as np
from torch.utils import data
from .parsers.atis import readATISFile


class PropheseeNCars(data.Dataset):
    """Prophesee N-Cars dataset from:
    Amos Sironi, Manuele Brambilla, Nicolas Bourdis, Xavier Lagorce, Ryad Benosman
    “HATS: Histograms of Averaged Time Surfaces for Robust Event-based Object Classification”.
    To appear in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018

    Available for download: https://www.prophesee.ai/2018/03/13/dataset-n-cars/
    """

    def __init__(self, path: str, is_train: bool = True, transforms=None):
        sub_path = "train" if is_train else "test"
        path = os.path.join(path, sub_path)
        assert os.path.exists(path)
        self._files = []
        self._labels = []

        for root, dirs, files in os.walk(path):
            label = os.path.basename(root)
            for file in files:
                if file.endswith(".dat"):
                    self._files.append(os.path.join(root, file))
                    self._labels.append(label)
        self._files = np.asarray(self._files)
        self._labels = np.asarray(self._labels)
        self.transforms = transforms

    def __len__(self):
        return self._labels.size

    def __getitem__(self, index):
        spike_train = readATISFile(self._files[index])
        spike_train.width = 120
        spike_train.height = 100
        spike_train.duration = 100000  # 100ms
        if self.transforms is not None:
            spike_train = self.transforms(spike_train)
        return spike_train, self._labels[index]
