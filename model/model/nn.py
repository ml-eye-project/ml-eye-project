import numpy as np
import tensorflow as tf

from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight

from .common import (
    COLOR_LABELS, Model, DictEncoder, IgnoreEncoder, SingleValueEncoder,
    ArrayEncoder, NormalizedEncoder, OneHotEncoder
)


class BalancedCategoricalAccuracy(keras.metrics.CategoricalAccuracy):
    def __init__(self, name='categorical_accuracy', dtype=None,
                 class_weight=None):
        self.class_weight = class_weight
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.class_weight is not None:
            y_true_int = tf.argmax(y_true, axis=1)
            sample_weight = tf.gather(self.class_weight, y_true_int)
        return super().update_state(y_true, y_pred, sample_weight=sample_weight)


class NeuralNetwork(Model):
    def __init__(self, config):
        self._input_encoder = DictEncoder([
            ('sex', SingleValueEncoder() if config['use_sex'] else IgnoreEncoder()),
            ('snp', ArrayEncoder(config['num_snps'], NormalizedEncoder(0, 2, -1, 1))),
        ])
        self._output_encoder = OneHotEncoder(len(COLOR_LABELS))

        model = keras.models.Sequential()

        model.add(keras.layers.Input(shape=(self._input_encoder.size,)))

        for i in range(3):
            size = config[f'l{i+1}_size']
            if not size:
                continue
            reg = config[f'l{i+1}_reg']
            model.add(keras.layers.Dense(size, activation='relu',
                kernel_regularizer=keras.regularizers.l2(reg)))

        model.add(keras.layers.Dense(
            self._output_encoder.size, activation='softmax'))

        opt = keras.optimizers.Adam(learning_rate=config['learning_rate'])
        acc = BalancedCategoricalAccuracy()
        model.compile(loss='categorical_crossentropy', optimizer=opt,
                      metrics=[acc])

        self._model = model
        self._accuracy = acc

    @property
    def input_encoder(self):
        return self._input_encoder

    @property
    def output_encoder(self):
        return self._output_encoder

    def _train(self, X_train, y_train, X_val, y_val):
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        y_train_int = np.argmax(y_train, axis=1)
        class_weight = compute_class_weight('balanced',
            classes=np.unique(y_train_int), y=y_train_int)
        class_weight_d = dict(enumerate(class_weight))

        self._accuracy.class_weight = class_weight

        history = self._model.fit(
            X_train, y_train, epochs=30, validation_data=(X_val, y_val),
            class_weight=class_weight_d, callbacks=early_stop)

        # EarlyStopping doesn't restore the best weights if we didn't exceed
        # the patience before the epoch limit, so do it manually in any case.
        self._model.set_weights(early_stop.best_weights)

        return {
            'metrics': {
                'train_loss': history.history['loss'],
                'train_acc': history.history['categorical_accuracy'],
                'val_loss': history.history['val_loss'],
                'val_acc': history.history['val_categorical_accuracy'],
            },
            'best_epoch': early_stop.best_epoch,
        }

    def _predict(self, X):
        return self._model.predict(X)
