import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold


SEX_LABELS = ['F', 'M']
SEX_LABEL_TO_IDX = {label: i for i, label in enumerate(SEX_LABELS)}

COLOR_LABELS = ['blue', 'brown', 'intermediate']
COLOR_LABEL_TO_IDX = {label: i for i, label in enumerate(COLOR_LABELS)}


class Dataset:
    def __init__(self, path):
        df = pd.read_csv(path)
        df.drop(columns=[
            'ID', 'class_pheno_2', 'collection', 'detailed_class'
        ], inplace=True)

        y = df.pop('class_pheno')
        sex = df.pop('sex')
        df.fillna(0, inplace=True)

        self.num_snps = df.shape[1]
        self.snp_names = list(df.columns)

        self.data = [
            {
                'x': {
                    'sex': SEX_LABEL_TO_IDX[row_sex],
                    'snp': row_snps.to_numpy(),
                },
                'y': COLOR_LABEL_TO_IDX[row_y],
            }
            for (_, row_snps), row_sex, row_y in zip(df.iterrows(), sex, y)
        ]

        # we produce balanced samples wrt color and sex
        self.stratify = y + sex

        # 70% training / 15% validation / 15% test
        # stratified sampling on color and sex
        strat = y + sex
        self.data_train, data_valtest, _, strat_valtest = train_test_split(
            self.data, strat, test_size=0.3, stratify=strat)
        self.data_val, self.data_test = train_test_split(data_valtest,
            test_size=0.5, stratify=strat_valtest)


    def iter_folds(self):
        # 66% train, 16.5% validation, 16.5% test
        kf_outer = StratifiedKFold(n_splits=3, shuffle=True)
        kf_inner = StratifiedKFold(n_splits=2, shuffle=True)

        X = np.zeros(len(self.data))

        for idxs_train, idxs_valtest in kf_outer.split(X, self.stratify):
            data_train = self.data[idxs_train]

            X_valtest = np.zeros(len(idxs_valtest))
            strat_valtest = self.stratify[idxs_valtest]
    
            for idxs_val, idxs_test in kf_inner.split(X_valtest, strat_valtest):
                data_val = self.data[idxs_val]
                data_test = self.data[idxs_test]

                yield data_train, data_val, data_test

    def iter_folds_train_test(self):
        # 80% train, 20% test
        kf = StratifiedKFold(n_splits=5, shuffle=True)
        X = np.zeros(len(self.data))
        for idxs_train, idxs_test in kf.split(X, self.stratify):
            data_train = [self.data[i] for i in idxs_train]
            data_test = [self.data[i] for i in idxs_test]
            yield data_train, data_test
