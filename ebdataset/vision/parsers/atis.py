import numpy as np
from ..type import DVSSpikeTrain


def readATISFile(filename: str) -> DVSSpikeTrain:
    with open(filename, "rb") as f_hndl:
        # Skip header
        while True:
            cursor = f_hndl.tell()
            if chr(f_hndl.readline()[0]) != "%":
                break

        if cursor >= 0:
            cursor += 2  # evType, evSize

        f_hndl.seek(cursor)

        # Read remaining bytes
        raw_data = np.fromfile(f_hndl, dtype=np.dtype("<u4"))

    timestamps = raw_data[0::2]
    positions = raw_data[1::2]

    data = DVSSpikeTrain(timestamps.size)
    data.x = positions & 0x3FFF
    data.y = np.right_shift(positions, 14) & 0x3FFF
    data.ps = np.right_shift(positions, 28)
    data.ts = timestamps

    return data
