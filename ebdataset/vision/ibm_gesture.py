import os
import re
from typing import List, Tuple, Union
import numpy as np
from torch.utils import data
import h5py
from tqdm import tqdm
from .parsers.aedat import readAEDATv3
from .type import DVSSpikeTrain


class IBMGesture(object):
    """IBM DVS Gesture dataset from
    A. Amir, B. Taba, D. Berg, T. Melano, J. McKinstry, C. Di Nolfo, T. Nayak, A. Andreopoulos, G. Garreau, M. Mendoza,
    J. Kusnitz, M. Debole, S. Esser, T. Delbruck, M. Flickner, and D. Modha,
    "A Low Power, Fully Event-Based Gesture Recognition System,"
    2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, 2017.

    Available for download at http://research.ibm.com/dvsgesture/

    Parsing is made with the AEDAT 3.1 format
    https://inivation.com/support4/software/fileformat/#aedat-31
    """

    _GESTURE_MAPPING_FILE = "gesture_mapping.csv"
    _TRAIN_TRIALS_FILE = "trials_to_train.txt"
    _TEST_TRIALS_FILE = "trials_to_test.txt"

    _LABELS_DTYPE = np.dtype(
        [
            ("event", np.uint8),
            ("start_time", np.uint32),  # In microsecond
            ("end_time", np.uint32),
        ]
    )
    _GESTURE_MAP = {}
    _TRAIN_FILES = []
    _TEST_FILES = []

    def __init__(self, path: str, shuffle: bool = True):
        """
        Arguments:
            path {str} -- The directory of the unzipped tarball containing the dataset

        Keyword Arguments:
            shuffle {bool} -- Shuffle the files before reading them. Note that the labels will
            still be ordered as presented in the files (default: {True})
        """
        assert os.path.exists(path), "The specified path doesn't exists"

        # Read gestures mapping file
        parsed_csv = np.genfromtxt(
            os.path.join(path, self._GESTURE_MAPPING_FILE),
            delimiter=",",
            skip_header=1,
            dtype=None,
            encoding="utf-8",
        )
        gestures, indexes = list(zip(*parsed_csv))
        self._GESTURE_MAP = dict(zip(indexes, gestures))

        # Read train trials file
        with open(os.path.join(path, self._TRAIN_TRIALS_FILE), "r") as f:
            self._TRAIN_FILES = map(lambda d: os.path.join(path, d.rstrip()), f.readlines())

        # Read test trials file
        with open(os.path.join(path, self._TEST_TRIALS_FILE), "r") as f:
            self._TEST_FILES = map(lambda d: os.path.join(path, d.rstrip()), f.readlines())

        self._TRAIN_FILES = list(filter(lambda f: os.path.isfile(f), self._TRAIN_FILES))
        self._TEST_FILES = list(filter(lambda f: os.path.isfile(f), self._TEST_FILES))

        if shuffle:
            np.random.shuffle(self._TRAIN_FILES)
            np.random.shuffle(self._TEST_FILES)

    def _read_labels(self, file: str) -> np.array:
        assert os.path.exists(file), "File %s doesn't exist" % file
        return np.genfromtxt(file, delimiter=",", skip_header=1, dtype=self._LABELS_DTYPE)

    def _parse_filename(self, file: str) -> Tuple[str, str, str]:
        trial = re.search(r"^user([0-9]+)_(.+)\.(aedat|csv)$", file, re.IGNORECASE)
        if trial:
            user, luminosity, file_type = trial.group(1, 2, 3)
            return (user, luminosity, file_type)
        return None

    def true_label(self, label_id: Union[str, int]) -> str:
        """Return the label class name for a label id

        Arguments:
            label_id {str} -- Label id as gotten from a sample

        Returns:
            str -- Label class name
        """
        return self._GESTURE_MAP[int(label_id)].rstrip()

    def _create_generator(self, files: List[str]):
        """Create a generator that yield samples over the array of files"""
        for file in files:
            labels = self._read_labels(file.replace(".aedat", "_labels.csv"))
            multilabel_spike_train = readAEDATv3(file)
            for (label_id, start_time, end_time) in labels:
                event_mask = (multilabel_spike_train.ts >= start_time) & (multilabel_spike_train.ts < end_time)
                ts = multilabel_spike_train.ts[event_mask] - start_time
                spike_train = DVSSpikeTrain(ts.size, width=128, height=128, duration=end_time - start_time + 1)
                spike_train.ts = ts
                spike_train.x = multilabel_spike_train.x[event_mask]
                spike_train.y = multilabel_spike_train.y[event_mask]
                spike_train.p = multilabel_spike_train.p[event_mask]
                yield spike_train, label_id

    def train_values_generator(self):
        """Create a generator iterating over the training samples
        The files are loaded in memory ad hoc
        Each sample is a tuple containing the spikes positions (x, y, polarity),
        the spike timing (in microsecond) and the label (int)"""
        return self._create_generator(self._TRAIN_FILES)

    def test_values_generator(self):
        """Create a generator iterating over the test samples
        The files are loaded in memory ad hoc
        Each sample is a tuple containing the spikes positions (x, y, polarity),
        the spike timing (in microsecond) and the label (int)"""
        return self._create_generator(self._TEST_FILES)

    def train_values(self):
        """Load and return the entire training dataset in memory
        Returns:
            np.array -- with shape (number of samples, 3) where the inner 3 represents:
            the spikes positions (x, y, polarity), the spike timing (in microsecond), and the label (int)
        """
        return np.array([i for i in self.train_values_generator()])

    def test_values(self):
        """Load and return the entire test dataset in memory
        Returns:
            np.array -- with shape (number of samples, 3) where the inner 3 represents:
            the spikes positions (x, y, polarity), the spike timing (in microsecond), and the label (int)
        """
        return np.array([i for i in self.test_values_generator()])


class H5IBMGesture(data.Dataset):
    """DVS Gesture dataset cached into a H5 file - Use H5DvsGesture.convert to create the h5 file"""

    _nb_of_samples = (1176, 288)  # in train, test
    _h5_prename = ("train", "test")
    _max_len = 19000000  # Recommended time padding (max duration of a sample)

    def __init__(self, path: str, is_train: bool = True):
        """path: location of the DvsGesture h5 file
        is_train: load training data
        """
        _, file_extension = os.path.splitext(path)
        if file_extension != ".h5":
            raise Exception("The dvs gesture must first be converted to a .h5 file. Please call H5DvsGesture.Convert")

        self.indx = 0 if is_train else 1
        self.file_path = path

    @staticmethod
    def convert(dvs_folder_path: str, h5_output_path: str, verbose=True):
        """dvs_folder_path : Path of the extracted tarball
        h5_output_path : Path of the output h5 file
        """

        _, file_extension = os.path.splitext(h5_output_path)
        if file_extension != ".h5":
            h5_output_path += ".h5"

        generator = IBMGesture(dvs_folder_path, shuffle=False)
        train_gen = generator.train_values_generator()
        test_gen = generator.test_values_generator()

        position_type = h5py.vlen_dtype(np.dtype("uint16"))
        time_type = h5py.vlen_dtype(np.dtype("uint32"))

        step_counter = tqdm(total=sum(H5IBMGesture._nb_of_samples), disable=(not verbose))

        with h5py.File(h5_output_path, "w-") as f:
            for (name, gen, length) in zip(
                H5IBMGesture._h5_prename,
                [train_gen, test_gen],
                H5IBMGesture._nb_of_samples,
            ):
                pos = f.create_dataset(name + "_pos", (length, 3), dtype=position_type)
                tos = f.create_dataset(name + "_tos", (length,), dtype=time_type)
                label = f.create_dataset(name + "_label", (length,), dtype=np.uint8)

                for i, (spike_train, label_id) in enumerate(gen):
                    pos[i, 0] = spike_train.x
                    pos[i, 1] = spike_train.y
                    pos[i, 2] = spike_train.p
                    tos[i] = spike_train.ts
                    label[i] = label_id
                    step_counter.update(1)

    def __len__(self):
        return self._nb_of_samples[self.indx]

    def __getitem__(self, index):
        if index >= self._nb_of_samples[self.indx]:
            raise StopIteration
        with h5py.File(self.file_path, "r") as file_hndl:
            name = self._h5_prename[self.indx]
            pos = file_hndl[name + "_pos"][index]
            tos = file_hndl[name + "_tos"][index]
            label = file_hndl[name + "_label"][index]

        spike_train = DVSSpikeTrain(tos.size, width=128, height=128, duration=tos.max() + 1)
        spike_train.x = pos[0]
        spike_train.y = pos[1]
        spike_train.p = pos[2]
        spike_train.ts = tos
        return spike_train, label
