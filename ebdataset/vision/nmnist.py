import os
import time
import numpy as np
from torch.utils import data
from .parsers.aer import readAERFile
from ..utils import download, unzip


class NMnist(data.Dataset):
    """
    NMnist dataset from
    Orchard, G.; Cohen, G.; Jayawant, A.; and Thakor, N.
    “Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades",
    Frontiers in Neuroscience, vol.9, no.437, Oct. 2015

    Available for download: https://www.garrickorchard.com/datasets/n-mnist
    """

    def __init__(self, path: str, is_train: bool = True, transforms=None, download_if_missing=True):
        if not os.path.exists(path) or len(os.listdir(path)) == 0:
            if download_if_missing:
                self._download_and_unzip(path)
            else:
                raise "Data not found at path %s" % path

        path = os.path.join(path, "Train" if is_train else "Test")

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

    def _download_and_unzip(self, output_directory):
        train_url = "https://www.dropbox.com/sh/tg2ljlbmtzygrag/AABlMOuR15ugeOxMCX0Pvoxga/Train.zip?dl=1"
        test_url = "https://www.dropbox.com/sh/tg2ljlbmtzygrag/AADSKgJ2CjaBWh75HnTNZyhca/Test.zip?dl=1"
        train_loc = os.path.join(output_directory, "Train%i.zip" % time.time())
        test_loc = os.path.join(output_directory, "Test%i.zip" % time.time())
        success = (
            download(train_url, train_loc, desc="Downloading training files")
            and unzip(train_loc, output_directory, desc="Extracting training files")
            and download(test_url, test_loc, desc="Downloading test files")
            and unzip(test_loc, output_directory, desc="Extracting test files")
        )

        if success:
            os.remove(train_loc)
            os.remove(test_loc)

        return success
