from sklearn import ensemble
from sklearn.metrics import accuracy_score

from .common import (
    Model, DictEncoder, IgnoreEncoder, SingleValueEncoder, ArrayEncoder,
    IdentityEncoder
)


class RandomForest(Model):
    def __init__(self, config):
        self._use_sex = config['use_sex']

        self._input_encoder = DictEncoder([
            ('sex', SingleValueEncoder() if self._use_sex else IgnoreEncoder()),
            ('snp', ArrayEncoder(config['num_snps'], SingleValueEncoder())),
        ])
        self._output_encoder = IdentityEncoder()

        self._clf = ensemble.RandomForestClassifier(
            config['num_estimators'],
            bootstrap=False,
            class_weight='balanced')

    def get_feature_importances(self, snp_names):
        feature_names = snp_names
        if self._use_sex:
            feature_names = ['sex'] + feature_names
        return {k: v for k, v in zip(
            feature_names, self._clf.feature_importances_)}

    @property
    def input_encoder(self):
        return self._input_encoder

    @property
    def output_encoder(self):
        return self._output_encoder

    def _train(self, X_train, y_train, X_val, y_val):
        self._clf.fit(X_train, y_train)

        y_train_pred = self._clf.predict(X_train)
        y_val_pred = self._clf.predict(X_val)
        return {
            'metrics': {
                'train_acc': [accuracy_score(y_train, y_train_pred)],
                'val_acc': [accuracy_score(y_val, y_val_pred)],
            },
            'best_epoch': 0,
        }

    def _predict(self, X):
        return self._clf.predict(X)
