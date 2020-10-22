import numpy as np
from quapy.error import tpr, fpr


def artificial_prevalence_sampling(n_samples=2100, smooth_limits_epsilon=0):
    return prevalence_linspace(n_prevalences=21, repeat=n_samples//21, smooth_limits_epsilon=smooth_limits_epsilon)


def natural_prevalence_sampling(n_samples, prev_mean, prev_std):
    return np.clip(np.random.normal(prev_mean, prev_std, n_samples), 0, 1)


def prevalence_linspace(n_prevalences=21, repeat=1, smooth_limits_epsilon=0.01):
    """
    Produces a uniformly separated values of prevalence. By default, produces an array 21 prevalences, with step 0.05
    and with the limits smoothed, i.e.:
    [0.01, 0.05, 0.10, 0.15, ..., 0.90, 0.95, 0.99]
    :param n_prevalences: the number of prevalence values to sample from the [0,1] interval (default 21)
    :param repeat: number of times each prevalence is to be repeated (defaults to 1)
    :param smooth_limits_epsilon: the quantity to add and subtract to the limits 0 and 1
    :return: an array of uniformly separated prevalence values
    """
    p = np.linspace(0., 1., num=n_prevalences, endpoint=True)
    p[0] += smooth_limits_epsilon
    p[-1] -= smooth_limits_epsilon
    if p[0] > p[1]:
        raise ValueError(f'the smoothing in the limits is greater than the prevalence step')
    if repeat > 1:
        p = np.repeat(p, repeat)
    return p


def prevalence_from_predictions(predictions, label=1):
    assert label in {0,1}, f'wrong label {label} requested for binary classification (+1,0)'
    predictions = 1-predictions if label == 0 else predictions
    return np.mean(predictions)


def prevalence_from_probabilities(posteriors, binarize: bool, label=1):
    assert label in {0, 1}, f'wrong label {label} requested for binary classification (+1,0)'
    predictions = posteriors>=0.5 if binarize else posteriors
    predictions = 1-predictions if label==0 else predictions
    return np.mean(predictions)


# def classifier_tpr_fpr(classfier_fn, validation_data: LabelledCollection):
def classifier_tpr_fpr(classfier_fn, validation_data):
    val_predictions = classfier_fn(validation_data.documents)
    return classifier_tpr_fpr_from_predictions(val_predictions, validation_data.labels)


def classifier_tpr_fpr_from_predictions(predictions, true_labels):
    tpr_ = tpr(true_labels, predictions)
    fpr_ = fpr(true_labels, predictions)
    return tpr_, fpr_


def adjusted_quantification(prevalence_estim, tpr, fpr, clip=True):
    den = tpr - fpr
    if den == 0:
        den += 1e-8
    adjusted = (prevalence_estim - fpr) / den
    if clip:
        adjusted = np.clip(adjusted, 0., 1.)
    return adjusted


