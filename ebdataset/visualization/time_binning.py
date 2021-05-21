"""Generate a time-binned plot from one of the available datasets
Usage: python -m ebdataset.visualization.time_binning --help
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
from ebdataset.vision import (
    INIRoshambo,
    H5IBMGesture,
    INIUCF50,
    NCaltech101,
    NMnist,
    PropheseeNCars,
)
from ebdataset.audio import NTidigits

assert __name__ == "__main__", "This script is meant to be run as main"

np.random.seed(0x1B)

parser = argparse.ArgumentParser()

available_datasets = [
    INIRoshambo,
    H5IBMGesture,
    INIUCF50,
    NCaltech101,
    NMnist,
    PropheseeNCars,
    NTidigits,
]
dataset_map = dict(zip([dataset.__name__ for dataset in available_datasets], available_datasets))

parser.add_argument("dataset", help="Dataset - One of [%s]" % " | ".join(dataset_map.keys()))
parser.add_argument("path", help="Path of the data directory or file for the chosen dataset")
parser.add_argument(
    "-i",
    "--id",
    help="Sample ID",
    type=int,
    default=-1,
)
parser.add_argument(
    "-b",
    "--bin_size",
    help="Size of the bin (ms)",
    type=float,
    default=1.0,
)

args = parser.parse_args()
dataset = dataset_map[args.dataset]
path = args.path

data_loader = dataset(path)
sample_id = np.random.randint(0, len(data_loader)) if args.id == -1 else args.id
spike_train, label = data_loader[sample_id]

duration = spike_train.ts.max() + 1

time_scale = getattr(spike_train, "time_scale", 1)
nb_bins = np.ceil(duration * time_scale * (1000.0 / args.bin_size)).astype(int)

fig, ax = plt.subplots()
ax.set_title(
    "Spike counts binned with size %.2f ms for dataset %s sample #%i" % (args.bin_size, args.dataset, sample_id)
)
ax.hist(spike_train.ts * time_scale, bins=nb_bins)
plt.show()
