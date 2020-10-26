import os
import pickle
import glob
from quapy.util import plot_diagonal
from absl import app
from app_helper import *
from pathlib import Path

flags.DEFINE_string('search_regex', None,
                    'regex indicating which predictions files (a pickle containing the true prevalences and the '
                    'estimated prevalences) are to be plotted')
flags.DEFINE_string('plot_path', '../plots/diagonal_plot.pdf', 'where to store the plot')
flags.DEFINE_string('title', 'Artificial Sampling Protocol', 'plot title')
#flags.mark_flags_as_required(['search_regex'])

FLAGS = flags.FLAGS

def main(_):
    estimations = {}
    for result in glob.glob(FLAGS.search_regex):
        print(result)
        #dataset, method, learner, length, error, run = Path(result).name.split('-')
        with open(result, 'rb') as fin:
            true_prevalences, estim_prevalences = pickle.load(fin)
            estimations[Path(result).name] = estim_prevalences

    if len(estimations) > 0:
        plot_diagonal(true_prevalences, estimations, savedir=FLAGS.plot_path, show=True)
    else:
        print(f'no matching found for regex={FLAGS.search_regex}')


if __name__ == '__main__':
    app.run(main)







