import numpy as np
import torch.utils.data as data
from quantities import hertz, ms, second


class ParityTask(data.IterableDataset):
    """Create a spike-based population encoded parity (or n-bits xor) task of max_iter samples
    if max_iter is not specified or is np.inf, this dataset will keep generating samples forever
    each HIGH bits is encoded with high_freq poisson-sampled spikes of shape features_per_bit x duration_per_bit
    LOW bits and background-noise is encoded with low_freq poisson-sampled spikes for the remaining of the sample_duration
    bits are encoded both temporally and spatially if sequential=True, otherwise only spatially
    """

    def __init__(
        self,
        seed=0x1B,
        low_freq=2 * hertz,
        high_freq=20 * hertz,
        sample_duration=2 * second,
        number_of_bits=2,
        features_per_bit=50,
        duration_per_bit=0.5 * second,
        dt=1 * ms,
        max_iter=np.inf,
        as_recarray=True,
        sequential=True,
    ):
        self.seed = seed
        self.max_iter = max_iter
        self.gen = np.random.RandomState(seed=self.seed)
        self.low_freq = float((low_freq * dt).simplified)
        self.high_freq = float((high_freq * dt).simplified)
        self.sample_duration = int((sample_duration / dt).simplified)
        self.duration_per_bit = int((duration_per_bit / dt).simplified)
        self.number_of_bits = number_of_bits
        self.features_per_bit = features_per_bit
        self.as_recarray = as_recarray
        self.sequential = sequential
        if sequential:
            assert (
                duration_per_bit * number_of_bits >= sample_duration
            ), "Sample duration is not enough to contain every bits"

    def __iter__(self):
        worker_info = data.get_worker_info()
        m = 1
        if worker_info is not None:  # multi-process data loading, re-seed the iterator
            self.gen = np.random.RandomState(seed=worker_info.id + self.seed)
            m = worker_info.num_workers

        i = 0
        while i < self.max_iter / m:
            i += 1

            bits = self.gen.randint(0, 2, size=self.number_of_bits)
            y = np.sum(bits) % 2

            spike_train = self.gen.poisson(
                lam=self.low_freq, size=(self.number_of_bits * self.features_per_bit, self.sample_duration)
            )

            for b in range(self.number_of_bits):
                if bits[b]:
                    time_pos = (
                        slice(b * self.duration_per_bit, (b + 1) * self.duration_per_bit)
                        if self.sequential
                        else slice(self.duration_per_bit)
                    )
                    spike_train[
                        b * self.features_per_bit : (b + 1) * self.features_per_bit, time_pos
                    ] = self.gen.poisson(lam=self.high_freq, size=(self.features_per_bit, self.duration_per_bit))

            if self.as_recarray:
                ts, addr = np.nonzero(spike_train.T)
                sample = np.recarray(shape=len(ts), dtype=[("addr", addr.dtype), ("ts", ts.dtype)])
                sample.addr = addr
                sample.ts = ts
                yield sample, y
            else:
                yield spike_train, y
