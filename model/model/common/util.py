import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay
)

from .dataset import COLOR_LABELS, SEX_LABELS


def seed_rngs():
    np.random.seed(42)
    tf.random.set_seed(42)


def train_model(model, data_train, data_val, plot=True):
    seed_rngs()

    X_train, y_train = _split_data(data_train)
    X_val, y_val = _split_data(data_val)
    history = model.train(X_train, y_train, X_val, y_val)

    metric_names = {
        'train_loss': 'Loss (training)',
        'train_acc': 'Accuracy (training)',
        'val_loss': 'Loss (validation)',
        'val_acc': 'Accuracy (validation)',
    }

    metrics = history['metrics']
    best_epoch = history['best_epoch']

    max_name_len = max(len(name) for name in metric_names.values())
    for key, name in metric_names.items():
        if key in metrics:
            print(f'{name:<{max_name_len}}: {metrics[key][best_epoch]:.2f}')

    if plot:
        df = pd.DataFrame(metrics)
        df.rename(columns=metric_names, inplace=True)
        ax = df.plot()
        ax.axvline(best_epoch, color='black', linestyle='--')
        plt.grid(True)

    return history


def evaluate_model(model, data, metrics=True, plot=True):
    seed_rngs()

    X_true, y_true = _split_data(data)
    y_pred = model.predict(X_true)

    cname = lambda y: [COLOR_LABELS[i] for i in y]
    y_true = cname(y_true)
    y_pred = cname(y_pred)

    print_test_accuracy(y_true, y_pred)

    class_metrics = metrics_by_class(y_true, y_pred)

    if metrics:
        print()
        print_classification_metrics(class_metrics)

    if plot:
        plot_confusion_matrix(y_true, y_pred)

    return class_metrics, y_true, y_pred


def print_test_accuracy(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    acc_bal = balanced_accuracy_score(y_true, y_pred)
    print(f'Test accuracy (unbalanced): {acc:.2f}')
    print(f'Test accuracy (balanced)  : {acc_bal:.2f}')


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=COLOR_LABELS,
                          normalize='true')
    disp = ConfusionMatrixDisplay(cm,
        display_labels=list(map(str.capitalize, COLOR_LABELS)))
    disp.plot(cmap='YlOrRd')


def show_plots():
    plt.show()


def print_classification_metrics(class_metrics):
    metrics_order = ['ppv', 'npv', 'tpr', 'tnr', 'f1', 'support']
    metrics_labels = {
        'ppv': 'PPV',
        'npv': 'NPV',
        'tpr': 'Sensitivity',
        'tnr': 'Specificity',
        'f1': 'F1 score',
        'support': 'Support',
    }

    class_metrics_df = pd.DataFrame(class_metrics)
    class_metrics_df = class_metrics_df[COLOR_LABELS]
    class_metrics_df = class_metrics_df.reindex(index=metrics_order)
    class_metrics_df.rename(
        index=metrics_labels,
        columns=lambda x: x.capitalize(),
        inplace=True)
    with pd.option_context('display.precision', 2):
        print(class_metrics_df)

    print()

    aggr_metrics = aggregate_metrics(class_metrics)
    aggr_metrics_df = pd.DataFrame(aggr_metrics)
    aggr_metrics_df = aggr_metrics_df[['avg', 'w_avg']]
    aggr_metrics_df = aggr_metrics_df.reindex(index=metrics_order)
    aggr_metrics_df.rename(
        index=metrics_labels,
        columns={
            'avg': 'Average',
            'w_avg': 'Weighted average',
        },
        inplace=True
    )
    with pd.option_context('display.precision', 2):
        print(aggr_metrics_df)


def metrics_by_class(y_true, y_pred):
    prec, rec, fscore, supp = precision_recall_fscore_support(y_true, y_pred,
        labels=COLOR_LABELS)
    npv = npv_score(y_true, y_pred)
    tnr = tnr_score(y_true, y_pred)

    metrics = {}
    for i, color in enumerate(COLOR_LABELS):
        metrics[color] = {
            'ppv': prec[i],
            'npv': npv[color],
            'tpr': rec[i],
            'tnr': tnr[color],
            'f1': fscore[i],
            'support': supp[i],
        }

    return metrics


def average_metrics(class_metrics_list):
    metric_names = list(class_metrics_list[0][COLOR_LABELS[0]].keys())
    avg_metrics = {}
    for color in COLOR_LABELS:
        avg_metrics[color] = {}
        for metric_name in metric_names:
            s = sum(elm[color][metric_name] for elm in class_metrics_list)
            avg_metrics[color][metric_name] = s / len(class_metrics_list)
    return avg_metrics


def aggregate_metrics(class_metrics):
    metric_names = list(class_metrics[COLOR_LABELS[0]].keys())

    metrics = {
        'avg': {},
        'w_avg': {},
    }

    for metric in metric_names:
        values = [class_metrics[color][metric] for color in COLOR_LABELS]
        if metric == 'support':
            metrics['avg'][metric] = metrics['w_avg'][metric] = sum(values)
        else:
            supps = [class_metrics[color]['support'] for color in COLOR_LABELS]
            metrics['avg'][metric] = sum(values) / len(values)
            metrics['w_avg'][metric] = sum(
                v * s for v, s in zip(values, supps)) / sum(supps)

    return metrics


def npv_score(y_true, y_pred):
    score = {}
    for color in COLOR_LABELS:
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt != color and yt == yp)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == color and yp != color)
        score[color] = tn / (tn + fn)
    return score


def tnr_score(y_true, y_pred):
    score = {}
    for color in COLOR_LABELS:
        neg = sum(1 for yt in y_true if yt != color)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt != color and yt == yp)
        score[color] = tn / neg
    return score


def _split_data(data):
    X = np.array([elm['x'] for elm in data])
    y = np.array([elm['y'] for elm in data])
    return X, y


def print_dataset_info(ds):
    subsets = {
        'Training': ds.data_train,
        'Validation': ds.data_val,
        'Test': ds.data_test,
    }

    colors_cap = [color.capitalize() for color in COLOR_LABELS]
    sex_ext = [{'F': 'Female', 'M': 'Male'}[sex] for sex in SEX_LABELS]

    counts = []
    for _, subset in subsets.items():
        sub_counts = {k: 0 for k in colors_cap + sex_ext}
        for example in subset:
            color = colors_cap[example['y']]
            sex = sex_ext[example['x']['sex']]
            sub_counts[color] += 1
            sub_counts[sex] += 1
        sub_counts['Total'] = len(subset)
        counts.append(sub_counts)

    df = pd.DataFrame(counts, index=subsets.keys())
    df.loc['Total'] = df.sum()
    print(df)

    counts = [{k: 0 for k in colors_cap} for _ in sex_ext]
    for example in ds.data:
        color = colors_cap[example['y']]
        sex_idx = example['x']['sex']
        counts[sex_idx][color] += 1

    df = pd.DataFrame(counts, index=sex_ext)
    print(df)
