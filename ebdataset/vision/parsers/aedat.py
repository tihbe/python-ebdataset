import os
import logging
import numpy as np
from ..type import DVSSpikeTrain

_logger = logging.getLogger(__name__)


def readAEDATv2_davies(file: str) -> np.recarray:
    """
    Parsing is made with the AEDAT 2.0 format for the davies camera
    https://inivation.com/support4/software/fileformat/#aedat-20
    with polarity events as packet data types

    Arguments:
        file {str} -- Complete path to file

    Returns:
        {tuple} -- A tuple of spike data and timestamp.
        The spike data is an array of x, y, and polarity values
    """
    assert os.path.exists(file), "File %s doesn't exist." % file

    with open(file, "rb") as f:
        version = f.readline()
        while True:  # Loop over header lines
            header_part = f.readline()
            is_a_comment = True
            if chr(header_part[0]) == "#":
                try:  # Bad luck that the first packets can start with ord('#')
                    header_part.decode("ascii")
                except (UnicodeDecodeError, AttributeError):
                    is_a_comment = False
            else:
                is_a_comment = False

            if not is_a_comment:  # If this isn't a comment, seek back to start of line
                f.seek(-len(header_part), 1)
                break
        data = f.read(-1)

    assert b"#!AER-DAT2.0" in version, "Unsupported data format detected"

    nbytes = len(data)
    count = -1
    if nbytes % 8 != 0:
        _logger.warning("Partial packet detected -- attempting to correct")
        count = nbytes // 8
    packets = np.frombuffer(data, dtype=np.dtype(">u8"), count=count)
    types = np.right_shift(packets, 63)
    if not np.all(types == 0):
        _logger.warning("All packets aren't from a DVS Camera (PS or IMU)")
        packets = packets[types == 0]

    data = DVSSpikeTrain(packets.size)
    data.y = np.bitwise_and(np.right_shift(packets, 54), 0x1FF)
    data.y = (
        np.max(data.y) - data.y
    )  # Lower left to upper left corner coordinate system
    data.x = np.bitwise_and(np.right_shift(packets, 44), 0x3FF)
    data.p = np.bitwise_and(np.right_shift(packets, 42), 0b11)
    data.ts = np.bitwise_and(packets, (1 << 32) - 1)
    return data


_AEDATV3_HEADER_DTYPE = np.dtype(
    [
        ("eventType", np.uint16),
        ("eventSource", np.uint16),
        ("eventSize", np.uint32),
        ("eventTSOffset", np.uint32),
        ("eventTSOverflow", np.uint32),
        ("eventCapacity", np.uint32),
        ("eventNumber", np.uint32),
        ("eventValid", np.uint32),
    ]
)
_AEDATV3_EVENT_DTYPE = np.dtype(
    [("fdata", np.uint32), ("timestamp", np.uint32),]  # In microsecond
)


def readAEDATv3(file: str) -> np.recarray:
    """
    Parsing is made with the AEDAT 3.1 format
    https://inivation.com/support4/software/fileformat/#aedat-31
    with polarity events as packet data types

    Arguments:
        file {str} -- Complete path to file

    Returns:
        {tuple} -- A tuple of spike data and timestamp.
        The spike data is an array of x, y, and polarity values
    """
    assert os.path.exists(file), "File %s doesn't exist." % file
    with open(file, "rb") as f:
        f_bytes = f.read()
    header, _, data = f_bytes.partition(b"\r\n#!END-HEADER\r\n")
    assert data, "Error loading data from file %s" % file
    version, data_format, source, date, *_ = header.split(b"\r\n")
    assert version == b"#!AER-DAT3.1", "Unsupported data format detected"
    offset = 0
    packets = []
    while offset < len(data):
        packet_header = np.frombuffer(
            data, dtype=_AEDATV3_HEADER_DTYPE, count=1, offset=offset
        )
        offset += packet_header.nbytes
        assert (
            packet_header["eventNumber"]
            == packet_header["eventCapacity"]
            == packet_header["eventValid"]
        ), "Something went wrong parsing the event header; your data might be corrupted"
        assert (
            packet_header["eventSize"] == _AEDATV3_EVENT_DTYPE.itemsize
        ), "Packet size doesn't correspond to underlying datatype"
        assert packet_header["eventType"] == 1  # Polarity events
        nb_packets = int(packet_header["eventNumber"])
        packets_data = np.frombuffer(
            data, dtype=_AEDATV3_EVENT_DTYPE, count=nb_packets, offset=offset
        )
        offset += packets_data.nbytes
        packets += packets_data.tolist()

    fdatas, timestamps = np.array(packets).T
    data = DVSSpikeTrain(fdatas.size)
    data.x = np.bitwise_and(np.right_shift(fdatas, 17), 0x7FFF)
    data.y = np.bitwise_and(np.right_shift(fdatas, 2), 0x7FFF)
    data.p = np.bitwise_and(np.right_shift(fdatas, 1), 0x1)
    data.ts = timestamps
    return data
