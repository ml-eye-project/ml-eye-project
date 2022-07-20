#!/usr/bin/env python3

import argparse

from collections import namedtuple
from distutils.util import strtobool

from model import NeuralNetwork, RandomForest
from model.common import (
    Dataset, seed_rngs, train_model, evaluate_model, show_plots,
    average_metrics, print_classification_metrics, plot_confusion_matrix,
    print_test_accuracy, print_dataset_info
)


ModelDesc = namedtuple('ModelDesc', ['model_class', 'config_desc'])
ConfigKeyDesc = namedtuple('ConfigKeyDesc', ['type', 'default', 'search_space'])


MODELS = {
    'forest': ModelDesc(RandomForest, {
        'use_sex': ConfigKeyDesc(strtobool, True, [True, False]),
        'num_estimators': ConfigKeyDesc(int, 100, [50, 100, 150, 200]),
    }),
    'nn': ModelDesc(NeuralNetwork, {
        'use_sex': ConfigKeyDesc(strtobool, True, [True, False]),
        'learning_rate': ConfigKeyDesc(float, 0.005, [0.001, 0.005, 0.01]),
        'l1_size': ConfigKeyDesc(int, 64, [32, 64]),
        'l1_reg': ConfigKeyDesc(float, 0.05, [0.0, 0.01, 0.05]),
        'l2_size': ConfigKeyDesc(int, 64, [32, 64]),
        'l2_reg': ConfigKeyDesc(float, 0.0, [0.0, 0.01, 0.05]),
        'l3_size': ConfigKeyDesc(int, 32, [32, 64]),
        'l3_reg': ConfigKeyDesc(float, 0.01, [0.0, 0.01, 0.05]),
    }),
}


def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate models.')

    parser.add_argument('model_name', choices=list(MODELS.keys()), help='model to use')
    parser.add_argument('dataset_path', metavar='dataset', help='path to dataset')

    parser.add_argument('--kfold', action='store_true', help='use K-Fold')

    parser.add_argument('--tune', action='store_true', help='tune hyperparameters')

    parser.add_argument('-c', '--config', metavar='KEY=VALUE', nargs='+',
                        default=[], help='set configuration value')

    parser.add_argument('--importance', action='store_true',
                        help='print feature importance')

    args = parser.parse_args()

    config = {}
    config_desc = MODELS[args.model_name].config_desc
    for kvs in args.config:
        k, v = kvs.split('=', maxsplit=1)
        config[k] = config_desc[k].type(v)
    args.config = config

    return args


def make_default_config(config_desc):
    return {key: desc.default for key, desc in config_desc.items()}


def count_tune_configs(config_desc, base_config):
    count = 1
    for key, desc in config_desc.items():
        if key not in base_config:
            count *= len(desc.search_space)
    return count


def iter_tune_configs(config_desc, base_config):
    tune_key = None
    for key in config_desc.keys():
        if key not in base_config:
            tune_key = key
            break

    if tune_key is None:
        yield base_config
        return

    for value in config_desc[tune_key].search_space:
        config = base_config.copy()
        config[tune_key] = value
        yield from iter_tune_configs(config_desc, config)


def tune(model_class, config_desc, base_config, data_train, data_val):
    print('[*] Tuning hyperparameters')

    num_configs = count_tune_configs(config_desc, base_config)

    i = 1
    best_config = None
    best_score = None
    best_model = None
    for config in iter_tune_configs(config_desc, base_config):
        print(f'[{i}/{num_configs}] Configuration: {config}')
        m = model_class(config)
        history = train_model(m, data_train, data_val, plot=False)
        score = history['metrics']['val_acc'][history['best_epoch']]
        if best_score is None or score > best_score:
            print(f'[+] Found better configuration, accuracy = {score:.2f}')
            best_config = config
            best_score = score
            best_model = m
        i += 1

    print(f'[+] Best configuration, accuracy {best_score:.2f}: {best_config}')

    return best_model


def train(model_class, config, data_train, data_val):
    print('[*] Training')
    m = model_class(config)
    train_model(m, data_train, data_val)
    return m


def train_kfold(model_class, config, ds):
    fold = 1
    fold_metrics = []
    y_true_all = []
    y_pred_all = []
    for data_train, data_test in ds.iter_folds_train_test():
        print(f'[*] Fold {fold}')
        m = model_class(config)
        train_model(m, data_train, data_train, plot=False) # TODO dummy validation
        metrics, y_true, y_pred = evaluate_model(m, data_test, metrics=False,
                                                 plot=False)
        fold_metrics.append(metrics)
        y_true_all += y_true
        y_pred_all += y_pred
        fold += 1

    print('[*] Averaging metrics across folds')
    avg_metrics = average_metrics(fold_metrics)

    print_test_accuracy(y_true_all, y_pred_all)
    print()
    print_classification_metrics(avg_metrics)
    plot_confusion_matrix(y_true_all, y_pred_all)
    show_plots()


def evaluate(m, data_test):
    print('[*] Evaluating')
    evaluate_model(m, data_test)
    show_plots()


def print_feature_importances(m, snp_names):
    print('[*] Feature importances:')
    imp = m.get_feature_importances(snp_names)
    print(imp)


def load_dataset(path):
    print('[*] Loading dataset')
    ds = Dataset(path)
    print_dataset_info(ds)
    return ds


def main():
    args = parse_args()
    model_desc = MODELS[args.model_name]

    seed_rngs()

    ds = load_dataset(args.dataset_path)

    base_config = {
        'num_snps': ds.num_snps,
    }
    base_config.update(args.config)

    def_config = make_default_config(model_desc.config_desc)
    def_config.update(base_config)

    if args.kfold:
        train_kfold(model_desc.model_class, def_config, ds)
    else:
        if args.tune:
            m = tune(model_desc.model_class, model_desc.config_desc,
                    base_config, ds.data_train, ds.data_val)
        else:
            m = train(model_desc.model_class, def_config, ds.data_train,
                    ds.data_val)
        print()
        evaluate(m, ds.data_test)

    if args.importance:
        print()
        print_feature_importances(m, ds.snp_names)


if __name__ == '__main__':
    main()
