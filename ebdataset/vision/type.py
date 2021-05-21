import numpy as np

_dtype = np.dtype([("x", np.uint16), ("y", np.uint16), ("p", np.bool_), ("ts", np.uint64)])


class DVSSpikeTrain(np.recarray):
    """Common type for event based vision datasets"""

    __name__ = "SparseVisionSpikeTrain"

    def __new__(cls, nb_of_spikes, *args, width=-1, height=-1, duration=-1, time_scale=1e-6, **nargs):
        obj = super(DVSSpikeTrain, cls).__new__(cls, nb_of_spikes, dtype=_dtype, *args, **nargs)
        obj.width = width
        obj.height = height
        obj.duration = duration
        obj.time_scale = time_scale  # dt duration in seconds

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self.width = getattr(obj, "width", None)
        self.height = getattr(obj, "height", None)
        self.duration = getattr(obj, "duration", None)
        self.time_scale = getattr(obj, "time_scale", None)
