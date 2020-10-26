import glob
import pickle
from scipy.stats import ttest_ind_from_stats, ttest_rel
from quapy.error import mae, mrae, ae, RAE
from collections import defaultdict
import numpy as np
import itertools


def __clean_name(method_name, del_run=True):
    #method_name is <path>/dataset_method_learner_length_optim_run.pkl
    method_name = method_name.lower()
    method_name = method_name.replace('.pkl', '')
    if '/' in method_name:
        method_name = method_name[method_name.rfind('/') + 1:]
    if del_run and '-run' in method_name:
        method_name = method_name[:method_name.find('-run')]
    return method_name


def evaluate_directory(result_path_regex='../results/*.pkl', evaluation_measures=[mae, mrae]):
    """
    A method that pre-loads all results and evaluates them in terms of some evaluation measures
    :param result_path_regex: the regular expression accepting all methods to be evaluated
    :param evaluation_measures: the evaluation metrics (a list of callable functions) to apply (e.g., mae, mrae)
    :return: a dictionary with keys the names of the methods plus a suffix -eval, and values the score of the
            evaluation metric (eval)
    """
    result_dict = defaultdict(lambda: [])

    for result in glob.glob(result_path_regex):
        with open(result, 'rb') as fin:
            true_prevalences, estim_prevalences = pickle.load(fin)

        method_name = __clean_name(result)
        for eval in evaluation_measures:
            score = eval(true_prevalences, estim_prevalences)
            result_dict[method_name+'-'+eval.__name__].append(score)

    result_dict = {method: np.mean(scores) for method, scores in result_dict.items()}
    return result_dict


def statistical_significance(result_path_regex='../results/*.pkl', eval_measure=ae):
    """
    Performs a series of two-tailored t-tests comparing any method with the best one found for the metric-dataset pair.
    :param result_path_regex: a regex to search for all methods that have to be submitted to the test
    :param eval_measure: the evaluation metric (e.g., ae, or rae) that will be the object of study of the test
    :return: a dictionary with keys the names of the methods, and values a tuple (x,y), i, which:
        x takes on values (best, verydifferent, different, nondifferent) indicating the outcome of the test w.r.t. the
            best performing method, for confidences pval<0.001, 0.001<=p-val<0.05, >=0.05, respectively
        y the interpolated rank, with 1 being assigned to the best method, 0 to the worst
    """
    result_dict = defaultdict(lambda: [])

    for result in glob.glob(result_path_regex):
        with open(result, 'rb') as fin:
            true_prevalences, estim_prevalences = pickle.load(fin)

        method_name = __clean_name(result)

        scores = eval_measure(true_prevalences, estim_prevalences)
        result_dict[method_name].extend(scores)

    method_score = [(method, np.mean(scores)) for method, scores in result_dict.items()]
    method_score = sorted(method_score, key=lambda x:x[1])
    best_method, mean1 = method_score[0]
    worst_method, meanworst = method_score[-1]
    std1, nobs1 = np.mean(result_dict[best_method]), len(result_dict[best_method])

    stats = {}
    for method, scores in result_dict.items():
        if method == best_method:
            stats[method] = ('best', 1)
        else:
            mean2 = np.mean(scores)
            std2  = np.std(scores)
            nobs2 = len(scores)
            _, pval = ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2)
            rel_rank = 1 - (mean2-mean1) / (meanworst-mean1)
            #_, pval = ttest_rel(best_scores, scores)
            if pval < 0.001:
                stats[method] = ('verydifferent', rel_rank)
            elif pval < 0.05:
                stats[method] = ('different', rel_rank)
            else:
                stats[method] = ('nondifferent', rel_rank)

    return stats


def statistical_significance_CC(method:str, eval_measure:callable, datasets:list, learners:list, optims:list, length=500):
    """
    Computes statistical tests between pairs of optimization methods (none, acce, f1e, mae) across datasets and
    learners, for a given method (e.g., cc) and an evaluation measure (e.g., rae)
    :param method: a cc-variant, subject of the statistical study
    :param eval_measure: an evaluation metric, the score being subjected to statistical comparison
    :param datasets: the datasets across which to perform the analysis
    :param learners: the learners across which to perform the analysis
    :param optims: the optimization metrics defining the samples to be compared
    :param length: length of the samples that generated the results
    :return: a dictionary with entries X-Y, where X and Y are two optimization metrics, and values S, where S is a
    symbol informing of the outcome of the test, according to:
       >> : method X is better than Y with a p-value < 0.001
       >  : method X is better than Y with a p-value < 0.05
       \sim: method X is not statistically significantly different from method Y, i.e., p-value >= 0.05
       << : method X is worse than Y with a p-value < 0.001
       <  : method X is worse than Y with a p-value < 0.05
    """
    runs = set()
    result_dict = defaultdict(lambda: [])

    def search_case_insensitive(method):
         return glob.glob(f'../results/*-{method.upper()}-*-500-*-run?.pkl') + \
                glob.glob(f'../results/*-{method.lower()}-*-500-*-run?.pkl')

    for result in search_case_insensitive(method):
        with open(result, 'rb') as fin:
            true_prevalences, estim_prevalences = pickle.load(fin)
        method_name = __clean_name(result, del_run=False)
        runs.add(method_name.split("run")[1])
        scores = eval_measure(true_prevalences, estim_prevalences)
        result_dict[method_name].extend(scores)
    if len(result_dict)==0:
        raise ValueError(f'error: regex produced no resuls')
    print(f'runs={runs}')

    combined_results = defaultdict(lambda: [])
    for optim, dataset, learner, run in itertools.product(optims, datasets, learners, runs):
        key=f'{dataset}-{method}-{learner}-{length}-{optim}-run{run}'
        if key not in result_dict:
            raise ValueError(f'missing {key}')
        combined_results[optim].extend(result_dict[key])

    ttest_comparisons = {}
    def means_symbol(mean1, mean2, pval):
        if mean1 < mean2:
            if pval < 0.001: return '$\gg$'
            if pval < 0.05: return '$>$'
            return '$\sim$'
        if mean1 == mean2: return '='
        if mean1 > mean2:
            if pval < 0.001: return '$\ll$'
            if pval < 0.05: return '$<$'
            return '$\sim$'

    for i,optim_i in enumerate(optims):
        for j,optim_j in enumerate(optims):
            if i==j:
                ttest_comparisons[f'{optim_i}-{optim_j}'] = '-'
                continue
            scores_i = combined_results[optim_i]
            scores_j = combined_results[optim_j]
            _, pval = ttest_rel(scores_i, scores_j)
            mean_i = np.mean(scores_i)
            mean_j = np.mean(scores_j)
            ttest_comparisons[f'{optim_i}-{optim_j}'] = means_symbol(mean_i, mean_j, pval)#+stat_symbol(pval)

    return ttest_comparisons


if __name__ == '__main__':
    # testing
    result_dict = evaluate_directory('../results/*.pkl', [mae])
    for method, scores in result_dict.items():
        print(f'{method}:={scores}')

    result_dict = statistical_significance('../results/*.pkl')
    for method, scores in result_dict.items():
        print(f'{method}:={scores}')

    methods = ['cc', 'acc', 'pcc', 'pacc']
    metrics = [ae, RAE]
    optims=['none', 'acce', 'f1e', 'mae']
    datasets=['imdb','kindle','hp']
    learners = ['svm', 'lr', 'mnb', 'rf', 'cnn']
    ss = {ccx: {m:statistical_significance_CC(ccx, m, datasets, learners, optims) for m in metrics} for ccx in methods}

    for i, optim_i in enumerate(optims):
        for optim_j in optims[i+1:]:
            pair=f'{optim_i}-{optim_j}'
            print(f'{pair}\t', end='')
            for method in methods:
                for metric in metrics:
                    print(ss[method][metric][pair]+'\t', end='')
            print()




