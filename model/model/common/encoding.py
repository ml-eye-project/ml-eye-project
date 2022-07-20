import numpy as np

from abc import ABC, abstractmethod


class Encoder(ABC):
    @abstractmethod
    def encode(self, value):
        pass

    @abstractmethod
    def decode(self, value):
        pass

    @property
    @abstractmethod
    def size(self):
        pass


class ArrayEncoder(Encoder):
    def __init__(self, size, elm_encoder):
        self._elm_encoder = elm_encoder
        self._size = size * elm_encoder.size

    def encode(self, value):
        return np.concatenate([self._elm_encoder.encode(x) for x in value])

    def decode(self, value):
        raise NotImplementedError()

    @property
    def size(self):
        return self._size


class DictEncoder(Encoder):
    def __init__(self, encoders):
        self._encoders = encoders
        self._size = sum(enc.size for _, enc in encoders)

    def encode(self, value):
        return np.concatenate([
            enc.encode(value[name])
            for name, enc in self._encoders
        ])

    def decode(self, value):
        raise NotImplementedError()

    @property
    def size(self):
        return self._size


class IgnoreEncoder(Encoder):
    def encode(self, value):
        return np.array([])

    def decode(self, value):
        raise NotImplementedError()

    @property
    def size(self):
        return 0


class IdentityEncoder(Encoder):
    def encode(self, value):
        return value

    def decode(self, value):
        return value

    @property
    def size(self):
        raise NotImplementedError()


class SingleValueEncoder(Encoder):
    def encode(self, value):
        return np.array([value])

    def decode(self, value):
        raise NotImplementedError()

    @property
    def size(self):
        return 1


class NormalizedEncoder(Encoder):
    def __init__(self, in_min, in_max, out_min, out_max):
        self._in_min = in_min
        self._out_min = out_min
        self._scale = (out_max - out_min) / (in_max - in_min)

    def encode(self, value):
        return np.array([
            (value - self._in_min) * self._scale + self._out_min
        ])

    def decode(self, value):
        raise NotImplementedError()

    @property
    def size(self):
        return 1


class OneHotEncoder(Encoder):
    def __init__(self, num_classes):
        self._num_classes = num_classes

    def encode(self, value):
        vec = np.zeros(self._num_classes)
        vec[int(value)] = 1
        return vec

    def decode(self, value):
        idx = np.argmax(value)
        return idx

    @property
    def size(self):
        return self._num_classes
