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
    result_dict = defaultdict(lambda: [])

    for result in glob.glob(result_path_regex):
        with open(result, 'rb') as fin:
            true_prevalences, estim_prevalences = pickle.load(fin)

        method_name = __clean_name(result)
        for eval in evaluation_measures:
            score = eval(true_prevalences, estim_prevalences)
            result_dict[method_name+'-'+eval.__name__].append(score)

    #for method, scores in result_dict.items():
    #    print(f'[ev]{method}: {len(scores)} {np.mean(scores)}')

    result_dict = {method: np.mean(scores) for method, scores in result_dict.items()}
    #print(sorted(result_dict.keys()))
    return result_dict


def statistical_significance(result_path_regex='../results/*.pkl', eval_measure=ae):
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
    regex_search = f'../results/*-{method}-*-500-*-run0.pkl'
    runs = set()
    result_dict = defaultdict(lambda: [])

    for result in glob.glob(regex_search):
        with open(result, 'rb') as fin:
            true_prevalences, estim_prevalences = pickle.load(fin)
        method_name = __clean_name(result, del_run=False)
        runs.add(method_name.split("run")[1])
        scores = eval_measure(true_prevalences, estim_prevalences)
        result_dict[method_name].extend(scores)
    if len(result_dict)==0:
        raise ValueError(f'error: regex "{regex_search}" produced no resuls')
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

    def stat_symbol(pval):
        if pval < 0.001: return '**'
        if pval < 0.05: return '*'
        return ''

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
    #result_dict = statistical_significance('../results/kindle-*.pkl')
    #for method, scores in result_dict.items():
    #    print(f'{method}:={scores}')
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




