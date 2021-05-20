import itertools
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from quapy.dataset.text import LabelledCollection
from quapy.method.aggregative import BaseQuantifier


def optimize_bert_for_quantification(method : BaseQuantifier,
                                devel_set: LabelledCollection,
                                error,
                                sample_size,
                                sample_prevalences,
                                param_grid,
                                n_jobs=-1):

    training, validation = devel_set.split_stratified(0.6)

    params_keys = list(param_grid.keys())
    params_values = list(param_grid.values())

    # generate the indexes that extract samples at desires prevalences
    sampling_indexes = [validation.sampling_index(prev, sample_size) for prev in sample_prevalences]

    # the true prevalences might slightly differ from the requested prevalences
    true_prevalences = np.array([validation.sampling_from_index(idx).prevalence() for idx in sampling_indexes])

    print(f'[starting optimization for BERT]')
    scores_params=[]
    for values in itertools.product(*params_values):
        params = {k: values[i] for i, k in enumerate(params_keys)}

        # overrides default parameters with the parameters being explored at this iteration
        method.set_params(**params)

        estim_prevalences = [method.quantify(validation.sampling_from_index(idx).documents) for idx in sampling_indexes]

        estim_prevalences = np.asarray(estim_prevalences)
        score = error(true_prevalences, estim_prevalences)
        print(f'checking hyperparams={params} got {error.__name__} score {score:.5f}')
        scores_params.append((score, params))
    scores, params = zip(*scores_params)
    best_pos = np.argmin(scores)
    best_params, best_score = params[best_pos], scores[best_pos]

    print(f'optimization finished: refitting for {best_params} (score={best_score:.5f}) on the whole development set')
    method.set_params(**best_params)


def optimize_for_quantification(method : BaseQuantifier,
                                devel_set: LabelledCollection,
                                error,
                                sample_size,
                                sample_prevalences,
                                param_grid,
                                n_jobs=-1):

    training, validation = devel_set.split_stratified(0.6)

    params_keys = list(param_grid.keys())
    params_values = list(param_grid.values())

    # generate the indexes that extract samples at desires prevalences
    sampling_indexes = [validation.sampling_index(prev, sample_size) for prev in sample_prevalences]

    # the true prevalences might slightly differ from the requested prevalences
    true_prevalences = np.array([validation.sampling_from_index(idx).prevalence() for idx in sampling_indexes])

    print(f'[starting optimization with n_jobs={n_jobs}]')
    scores_params=[]
    for values in itertools.product(*params_values):
        params = {k: values[i] for i, k in enumerate(params_keys)}

        # overrides default parameters with the parameters being explored at this iteration
        method.set_params(**params)
        method.fit(training)

        estim_prevalences = Parallel(n_jobs=n_jobs)(
            delayed(method.quantify)(validation.sampling_from_index(idx).documents) for idx in sampling_indexes
        )
        estim_prevalences = np.asarray(estim_prevalences)
        score = error(true_prevalences, estim_prevalences)
        print(f'checking hyperparams={params} got {error.__name__} score {score:.5f}')
        scores_params.append((score, params))
    scores, params = zip(*scores_params)
    best_pos = np.argmin(scores)
    best_params, best_score = params[best_pos], scores[best_pos]

    print(f'optimization finished: refitting for {best_params} (score={best_score:.5f}) on the whole development set')
    method.set_params(**best_params)
    method.fit(devel_set)


def optimize_for_classification(method : BaseQuantifier,
                                devel_set : LabelledCollection,
                                error,
                                param_grid):

    training, validation = devel_set.split_stratified(0.6)

    params_keys = list(param_grid.keys())
    params_values = list(param_grid.values())

    best_p, best_error = None, None
    pbar = tqdm(list(itertools.product(*params_values)))
    for values_ in pbar:
        params_ = {k: values_[i] for i, k in enumerate(params_keys)}

        # overrides default parameters with the parameters being explored at this iteration
        method.set_params(**params_)
        method.fit(training)
        class_predictions = method.classify(validation.documents)
        score = error(validation.labels, class_predictions)

        if best_error is None or score < best_error:
            best_error, best_p = score, params_
        pbar.set_description(
            f'checking hyperparams={params_} got got {error.__name__} score={score:.5f} [best params = {best_p} with score {best_error:.5f}]'
        )

    print(f'optimization finished: refitting for {best_p} on the whole development set')
    method.set_params(**best_p)
    method.fit(devel_set)