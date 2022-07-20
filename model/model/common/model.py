import numpy as np

from abc import ABC, abstractmethod


class Model(ABC):
    def train(self, X_train, y_train, X_val, y_val):
        X_train = self._encode_values(X_train, self.input_encoder)
        y_train = self._encode_values(y_train, self.output_encoder)
        X_val = self._encode_values(X_val, self.input_encoder)
        y_val = self._encode_values(y_val, self.output_encoder)
        return self._train(X_train, y_train, X_val, y_val)

    def predict(self, X):
        X = self._encode_values(X, self.input_encoder)
        y = self._predict(X)
        return self._decode_values(y, self.output_encoder)

    @abstractmethod
    def get_feature_importances(self, snp_names):
        pass

    @property
    @abstractmethod
    def input_encoder(self):
        pass

    @property
    @abstractmethod
    def output_encoder(self):
        pass

    @abstractmethod
    def _train(self, X_train, y_train, X_val, y_val):
        pass

    @abstractmethod
    def _predict(self, X):
        pass

    @staticmethod
    def _encode_values(values, encoder):
        return np.array([encoder.encode(x) for x in values])

    @staticmethod
    def _decode_values(values, encoder):
        return np.array([encoder.decode(x) for x in values])
