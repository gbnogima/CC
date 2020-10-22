import itertools
import multiprocessing
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt



def get_parallel_slices(n_tasks, n_jobs=-1):
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    batch = int(n_tasks / n_jobs)
    remainder = n_tasks % n_jobs
    return [slice(job * batch, (job + 1) * batch + (remainder if job == n_jobs - 1 else 0)) for job in
            range(n_jobs)]


def parallelize(func, args, n_jobs):
    slices = get_parallel_slices(len(args), n_jobs)

    results = Parallel(n_jobs=n_jobs)(
        delayed(func)(args[slice_i]) for slice_i in slices
    )
    return list(itertools.chain.from_iterable(results))


def plot_diagonal(prevalences, methods_predictions, train_prev=None, test_prev=None,
                  title='Artificial Sampling Protocol', savedir=None, show=True):
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 200
    methodnames, method_predictions = zip(*list(methods_predictions.items()))
    x_ticks = np.sort(np.unique(prevalences))

    ave = np.array([[np.mean(method_i[prevalences == p]) for p in x_ticks] for method_i in method_predictions])
    std = np.array([[np.std(method_i[prevalences == p]) for p in x_ticks] for method_i in method_predictions])

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.grid()
    ax.plot([0,1], [0,1], '--k', label='ideal', zorder=1)
    for i,method in enumerate(ave):
        label = methodnames[i]
        ax.errorbar(x_ticks, method, fmt='-', marker='o', label=label, markersize=3, zorder=2)
        ax.fill_between(x_ticks, method-std[i], method+std[i], alpha=0.25)
    if train_prev is not None:
        ax.scatter(train_prev, train_prev, c='c', label='tr-prev', linewidth=2, edgecolor='k', s=100, zorder=3)
    if test_prev is not None:
        ax.scatter(test_prev, test_prev, c='y', label='te-prev', linewidth=2, edgecolor='k', s=100, zorder=3)

    ax.set(xlabel='true prevalence', ylabel='estimated prevalence', title=title)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if savedir is not None:
        plt.savefig(savedir)

    if show:
        plt.show()


