"""Generate a video from one of the available vision datasets
Usage: python -m ebdataset.visualization.spike_train_to_vid --help
"""
import argparse
import cv2
import numpy as np
from ebdataset.vision import (
    INIRoshambo,
    H5IBMGesture,
    INIUCF50,
    NCaltech101,
    NMnist,
    PropheseeNCars,
)
from tqdm import tqdm

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
]
dataset_map = dict(zip([dataset.__name__ for dataset in available_datasets], available_datasets))

parser.add_argument("dataset", help="Dataset - One of [%s]" % " | ".join(dataset_map.keys()))
parser.add_argument("path", help="Path of the data directory or file for the chosen dataset")
parser.add_argument(
    "-n",
    "--num_samples",
    help="Number of video samples to generate",
    type=int,
    default=10,
)
parser.add_argument("-d", "--dilatation", help="Time dilatation scale", type=float, default=1.0)  # Default Real time
parser.add_argument("-s", "--scale", help="Spatial scaling", type=float, default=1.0)  # Default Real size

args = parser.parse_args()
dataset = dataset_map[args.dataset]
path = args.path

data_loader = dataset(path)
sample_idx = np.random.randint(0, len(data_loader), size=args.num_samples)

spatial_scale = args.scale
time_scale = args.dilatation

for i, sample_id in enumerate(tqdm(sample_idx)):
    spike_train, label = data_loader[sample_id]
    filename = "%s_%i_sample_%i_label_%s.avi" % (args.dataset, i, sample_id, str(label))
    out_width, out_height = (
        int(spike_train.width * spatial_scale),
        int(spike_train.height * spatial_scale),
    )
    out_duration = spike_train.duration * spike_train.time_scale * time_scale
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"MP42"), 60.0, (out_width, out_height))
    for frame_start in np.arange(0.0, out_duration, 1 / 60.0):
        frame_end = frame_start + 1 / 60.0
        ts = spike_train.ts * spike_train.time_scale * time_scale
        mask = (ts >= frame_start) & (ts < frame_end)
        frame = np.zeros((out_width, out_height, 3), dtype=np.uint8)
        for x, y, p in zip(spike_train.x[mask], spike_train.y[mask], spike_train.p[mask]):
            frame[
                int(x * spatial_scale) : int((x + 1) * spatial_scale),
                int(y * spatial_scale) : int((y + 1) * spatial_scale),
                int(p),
            ] = 255

        out.write(np.swapaxes(frame, 0, 1))
    out.release()
    print("File %s created." % filename)
