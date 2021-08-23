import numpy as np
from ..utils import mathutil
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from .epoch import Epoch
from .signal import Signal
from .datawriter import DataWriter


class Position(DataWriter):
    def __init__(
        self,
        traces: np.ndarray,
        t_start=0,
        sampling_rate=120,
        metadata=None,
        filename=None,
    ) -> None:

        if traces.ndim == 1:
            traces = traces.reshape(1, -1)

        assert traces.shape[0] <= 3, "Maximum possible dimension of position is 3"
        self.traces = traces
        self._t_start = t_start
        self._sampling_rate = sampling_rate
        self.metadata = metadata
        super().__init__(filename=filename)

    @property
    def x(self):
        return self.traces[0]

    @x.setter
    def x(self, x):
        self.traces[0] = x

    @property
    def y(self):
        assert self.ndim > 1, "No y for one-dimensional position"
        return self.traces[1]

    @y.setter
    def y(self, y):
        assert self.ndim > 1, "Position data has only one dimension"
        self.traces[1] = y

    @property
    def z(self):
        assert self.ndim == 3, "Position data is not three-dimensional"
        return self.traces[2]

    @z.setter
    def z(self, z):
        self.traces[2] = z

    @property
    def t_start(self):
        return self._t_start

    @t_start.setter
    def t_start(self, t):
        self._t_start = t

    @property
    def n_frames(self):
        return self.traces.shape[1]

    @property
    def duration(self):
        return self.n_frames / self.sampling_rate

    @property
    def t_stop(self):
        return self.t_start + self.duration

    @property
    def time(self):
        return np.linspace(self.t_start, self.t_stop, self.n_frames)

    @property
    def ndim(self):
        return self.traces.shape[0]

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, sampling_rate):
        self._sampling_rate = sampling_rate

    def to_dict(self):
        data = {
            "traces": self.traces,
            "t_start": self.t_start,
            "sampling_rate": self._sampling_rate,
            "metadata": self.metadata,
        }
        return data

    @staticmethod
    def from_dict(d):
        return Position(
            traces=d["traces"],
            t_start=d["t_start"],
            sampling_rate=d["sampling_rate"],
            metadata=d["metadata"],
        )

    @property
    def speed(self):
        self.speed = np.sqrt(np.diff(self.x) ** 2 + np.diff(self.y) ** 2) / (
            1 / self.sampling_rate
        )

    def to_dataframe(self):
        return pd.DataFrame(self.to_dict)

    def speed_in_epochs(self, epochs: Epoch):
        assert isinstance(epochs, Epoch), "epochs must be neuropy.Epoch object"
        pass

    def time_slice(self, t_start, t_stop):
        if t_start is None:
            t_start = self.t_start

        if t_stop is None:
            t_stop = self.t_stop

        indices = (self.time > t_start) & (self.time < t_stop)

        return Position(
            traces=self.traces[:, indices],
            t_start=t_start,
            sampling_rate=self.sampling_rate,
        )