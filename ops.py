import pandas as pd


def optimize_threshold(probabilities: pd.Series, labels: pd.Series, method: callable):
    def compute_score(th):
        y_pred = probabilities > th
        return method(y_pred=y_pred, y_true=labels)

    return max(probabilities.unique(), key=compute_score)


def nape_from_numbers(n_true, n_pred):
    return -abs(n_true - n_pred) / n_true


def nape_from_labels(y_true, y_pred):
    n_true = y_true.sum()
    n_pred = y_pred.sum()
    return nape_from_numbers(n_true, n_pred)


