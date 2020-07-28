import numpy as np
from ..type import DVSSpikeTrain


def readAERFile(filename: str) -> DVSSpikeTrain:
    """Function adapted from https://github.com/gorchard/event-Python/blob/master/eventvision.py
    for reading AER files from N-MNIST and N-Caltech 101
    """

    raw_data = np.fromfile(filename, dtype=np.uint8).astype(np.uint32)

    all_x = raw_data[0::5]
    all_y = raw_data[1::5]
    all_p = np.right_shift(raw_data[2::5], 7)
    all_ts = (
        np.left_shift(raw_data[2::5] & 127, 16)
        | np.left_shift(raw_data[3::5], 8)
        | raw_data[4::5]
    )

    # Process time stamp overflow events
    time_increment = 2 ** 13
    overflow_indices = np.where(all_y == 240)[0]
    for overflow_index in overflow_indices:
        all_ts[overflow_index:] += time_increment

    # Everything else is a proper td spike
    td_indices = np.where(all_y != 240)[0]

    data = DVSSpikeTrain(td_indices.size)

    data.x = all_x[td_indices]
    data.y = all_y[td_indices]
    data.ts = all_ts[td_indices]
    data.p = all_p[td_indices]
    return data
