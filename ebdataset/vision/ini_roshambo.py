import os
import numpy as np
from h5py import File
from tqdm import tqdm
from .parsers.aedat import readAEDATv2_davies
from torch.utils.data.dataset import Dataset
from .type import DVSSpikeTrain
from quantities import us


class INIRoshambo(Dataset):
    """INI Roshambo17 dataset from:
    I.-A. Lungu, F. Corradi, and T. Delbruck,
    Live Demonstration: Convolutional Neural Network Driven by Dynamic Vision Sensor Playing RoShamBo,
    in 2017 IEEE Symposium on Circuits and Systems (ISCAS 2017), Baltimore, MD, USA, 2017 [Online].
    Available: https://drive.google.com/file/d/0BzvXOhBHjRheYjNWZGYtNFpVRkU/view?usp=sharing

    Download available at:
    https://docs.google.com/document/d/e/2PACX-1vTNWYgwyhrutBu5GpUSLXC4xSHzBbcZreoj0ljE837m9Uk5FjYymdviBJ5rz-f2R96RHrGfiroHZRoH/pub
    """

    def __init__(self, path: str, with_backgrounds=False, transforms=None):
        """
        :param path: path of the aedat folder or h5 file (faster)
        :param transforms: torchvision-like transforms (optional)
        """
        assert os.path.exists(path)

        self.path = path

        if os.path.isdir(path):  # AEDat v2 directory
            self.backend = "aedat"
            self.samples = filter(lambda f: os.path.splitext(f)[1] == ".aedat", os.listdir(path))
        elif os.path.splitext(path)[1] == ".h5":
            self.backend = "h5"
            with File(path, "r", libver="latest") as f_hndl:
                self.samples = list(f_hndl.keys())

        if not with_backgrounds:
            self.samples = filter(lambda f: not ("background" in f), self.samples)

        self.samples = list(self.samples)
        self.transforms = transforms
        self.with_backgrounds = with_backgrounds

    def convert(self, out_path, verbose=False):
        """
        Converts a aedat directory to a h5 file for faster processing

        :param path: Aedat directory
        :param out_path:  Output h5 file
        :return: New Roshambo object with h5 file as backend
        """

        if self.backend == "h5":  # Send back object if we're already using an h5 backend
            return self

        if not (".h5" in out_path):
            out_path += ".h5"

        with File(out_path, "w-", libver="latest") as f_hndl:
            for sample_id in tqdm(self.samples, disable=not verbose):
                sparse_spike_train = readAEDATv2_davies(os.path.join(self.path, sample_id))
                sparse_spike_train.ts = sparse_spike_train.ts - np.min(sparse_spike_train.ts)  # Start the sample at t=0
                f_hndl[sample_id] = sparse_spike_train

        return INIRoshambo(out_path, with_backgrounds=self.with_backgrounds, transforms=self.transforms)

    def split_to_subsamples(self, out_path, duration_per_sample, verbose=False):
        if not (".h5" in out_path):
            out_path += ".h5"

        duration_per_sample = int(duration_per_sample.rescale(us).magnitude)

        with File(out_path, "w-", libver="latest") as f_hndl:
            for i, (sample, label) in enumerate(tqdm(self, disable=not verbose)):
                sample_id = self.samples[i]
                label_test, *_ = os.path.splitext(sample_id)[0].split("_")
                assert label_test == label  # Making sure there is no mix up
                total_duration = np.max(sample.ts) + 1
                for j, start_time in enumerate(
                    tqdm(
                        range(0, total_duration, duration_per_sample),
                        disable=not verbose,
                    )
                ):
                    if start_time + duration_per_sample > total_duration:  # End
                        break
                    sub_mask = (sample.ts >= start_time) & (sample.ts < start_time + duration_per_sample)
                    nb_of_spikes = np.sum(sub_mask)
                    if nb_of_spikes <= 10:
                        continue
                    sub_sample = DVSSpikeTrain(nb_of_spikes, duration=duration_per_sample)
                    sub_sample.ts = sample.ts[sub_mask]
                    sub_sample.ts = sub_sample.ts - np.min(sub_sample.ts)  # Start at 0
                    sub_sample.x = sample.x[sub_mask]
                    sub_sample.y = sample.y[sub_mask]
                    sub_sample.p = sample.p[sub_mask]
                    f_hndl[f"{sample_id}_{j}"] = sub_sample

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_id = self.samples[index]
        label, *extra_info = os.path.splitext(sample_id)[0].split("_")
        if self.backend == "aedat":
            filename = os.path.join(self.path, sample_id)
            sparse_spike_train = readAEDATv2_davies(filename)
            sparse_spike_train.ts = sparse_spike_train.ts - np.min(sparse_spike_train.ts)  # Start the sample at t=0
        elif self.backend == "h5":
            with File(self.path, "r", libver="latest") as f_hndl:
                sparse_spike_train = f_hndl[sample_id][()]
            sparse_spike_train = np.rec.array(sparse_spike_train, dtype=sparse_spike_train.dtype).view(DVSSpikeTrain)

        sparse_spike_train.width = 240
        sparse_spike_train.height = 180
        sparse_spike_train.duration = sparse_spike_train.ts.max() + 1
        sparse_spike_train.time_scale = 1e-6

        if self.transforms is not None:
            sparse_spike_train = self.transforms(sparse_spike_train)

        return sparse_spike_train, label
