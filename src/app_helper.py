import os
import pickle
from pathlib import Path
from absl import flags, logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import settings
import quapy.error as errors
from quapy.classification.neural import NeuralClassifierTrainer
from quapy.dataset.text import TQDataset
from quapy.method import NEURAL_METHODS
from quapy.method.aggregative import *
from quapy.method.non_aggregative import *
from quapy.optimization import *
from quapy.util import plot_diagonal

import bert
from bert import Bert

FLAGS = flags.FLAGS

# quantifiers:
# ----------------------------------------
# alias for quantifiers and default configurations
QUANTIFIER_ALIASES = {
    'cc': lambda learner: ClassifyAndCount(learner),
    'acc': lambda learner: AdjustedClassifyAndCount(learner),
    'pcc': lambda learner: ProbabilisticClassifyAndCount(learner),
    'pacc': lambda learner: ProbabilisticAdjustedClassifyAndCount(learner),
    'emq': lambda learner: ExpectationMaximizationQuantifier(learner),
    'svmq': lambda learner: SVMQ(settings.SVM_PERF_HOME),
    'svmkld': lambda learner: SVMKLD(settings.SVM_PERF_HOME),
    'svmnkld': lambda learner: SVMNKLD(settings.SVM_PERF_HOME),
    'svmmae': lambda learner: SVMAE(settings.SVM_PERF_HOME),
    'svmmrae': lambda learner: SVMRAE(settings.SVM_PERF_HOME),
    'hdy': lambda learner: HellingerDistanceY(learner),
    'quanet': lambda learner: QuaNet(learner, device=settings.TORCHDEVICE),
    'mlpe': lambda learner: MaximumLikelihoodPrevalenceEstimation(),
}


# learners:
# ----------------------------------------
TFIDF_BASED={'svm', 'lr', 'mnb', 'rf', 'svmperf', 'none'}
DEEPLEARNING_BASED={'cnn'}

# alias for classifiers/regressors and default configurations
LEARNER_ALIASES = {
    'svm': lambda: LinearSVC(),
    'lr': lambda: LogisticRegression(),
    'mnb': lambda: MultinomialNB(),
    'rf': lambda: RandomForestClassifier(),
    'svmperf': lambda: SVMperf(settings.SVM_PERF_HOME),
    'cnn': lambda: NeuralClassifierTrainer('cnn', FLAGS.vocabulary_size, device=settings.TORCHDEVICE),
    'bert': lambda: Bert(),
    'none': lambda: None
}

# hyperparameter spaces for each classifier/regressor
__C_range = np.logspace(-4, 5, 10)

HYPERPARAMS = {
    'svm': {'C': __C_range, 'class_weight': [None, 'balanced']},
    'lr': {'C': __C_range, 'class_weight': [None, 'balanced']},
    'mnb': {'alpha': np.linspace(0., 1., 21)},
    'rf': {'n_estimators': [10, 50, 100, 250, 500], 'max_depth': [5, 15, 30], 'criterion': ['gini', 'entropy']},
    'svmperf': {'C': __C_range},
    'cnn': {'embedding_size': [100, 300], 'hidden_size': [256, 512], 'drop_p': [0, 0.5], 'weight_decay': [0, 1e-4]},
    'none': {}
}


# apps' utils:
# ----------------------------------------

def load_dataset():
    logging.info(f'loading dataset {FLAGS.dataset}')
    train_path = f'{FLAGS.dataset}/train.txt'
    test_path = f'{FLAGS.dataset}/test.txt'
    return TQDataset.from_files(train_path, test_path)


def resample_training_prevalence(benchmark: TQDataset):
    prev = FLAGS.trainp
    if prev is None:
        return benchmark
    else:
        logging.info(f'resampling training set at p={100*FLAGS.trainp:.2f}%')
        assert 0 < prev < 1, f'error: trainp ({prev}) must be in (0,1)'
        new_training = benchmark.training.undersampling(prev)
        return TQDataset(training=new_training, test=benchmark.test)


def prepare_dataset(benchmark: TQDataset):
    if FLAGS.learner.lower() in TFIDF_BASED:
        benchmark.tfidfvectorize()
    elif FLAGS.learner.lower() in DEEPLEARNING_BASED:
        benchmark.index()
    elif FLAGS.learner.lower() == 'bert':
        bert.params = benchmark.bert_preprocessing()
    else:
        raise ValueError(f'unknown representation for learner {FLAGS.learner}')
    if hasattr(FLAGS, 'vocabulary_size'):
        FLAGS.vocabulary_size = len(benchmark.vocabulary_)
    else:
        flags.DEFINE_integer('vocabulary_size', len(benchmark.vocabulary_), '')
    return benchmark


def instantiate_learner():
    logging.info(f'instantiating classifier {FLAGS.learner}')

    learner = FLAGS.learner.lower()
    if learner not in LEARNER_ALIASES:
        raise ValueError(f'unknown learner {FLAGS.learner}')

    return LEARNER_ALIASES[learner]()


def instantiate_quantifier(learner):
    logging.info(f'instantiating quantifier {FLAGS.method}')

    method = FLAGS.method.lower()
    if method not in QUANTIFIER_ALIASES:
        raise ValueError(f'unknown quantification method {FLAGS.method}')
    
    return QUANTIFIER_ALIASES[method](learner)


def instantiate_error():
    logging.info(f'instantiating error {FLAGS.error}')
    return getattr(errors, FLAGS.error)


def model_selection(method, benchmark: LabelledCollection):
    learner = FLAGS.learner.lower()
    if learner == 'bert':
        logging.info('using BERT classifier (error set as none)')
        method.fit(benchmark.training)
    elif FLAGS.error != 'none':
        error = instantiate_error()
        optimization(method, error, benchmark.training)
    else:
        logging.info('using default classifier (no model selection will be performed)')
        method.fit(benchmark.training)


def save_results(true_prevalences, estim_prevalences):
    os.makedirs(FLAGS.results_path, exist_ok=True)
    fout = f'{FLAGS.results_path}/{run_name()}.pkl'
    logging.info(f'saving results in {fout}')
    true_prevalences = np.asarray(true_prevalences)
    estim_prevalences = np.asarray(estim_prevalences)
    with open(fout, 'wb') as foo:
        pickle.dump(tuple((true_prevalences, estim_prevalences)), foo, pickle.HIGHEST_PROTOCOL)


def run_name():
    dataset_name = Path(FLAGS.dataset).name
    suffix = f'-{FLAGS.suffix}' if FLAGS.suffix else ''
    return f'{dataset_name}-{FLAGS.method}-{FLAGS.learner}-{FLAGS.sample_size}-{FLAGS.error}' + suffix


def decide_njobs(method):
    n_jobs = -1
    if check_require_cuda(method):
        n_jobs = 1
        logging.warning(f'n_jobs was set to 1 since there seem to be applications running in GPU')
    return n_jobs


def check_require_cuda(method):
    learner = FLAGS.learner.lower()
    return (method.__class__ in NEURAL_METHODS or learner in DEEPLEARNING_BASED) and settings.TORCHDEVICE=='cuda'


def optimization(method, error, training):
    logging.info(f'exploring hyperparameters')

    learner = FLAGS.learner.lower()

    if error in errors.CLASSIFICATION_ERROR:
        logging.info(f'optimizing for classification [{error.__name__}]')
        optimize_for_classification(
            method,
            training,
            error,
            param_grid=HYPERPARAMS[learner]
        )
    elif error in errors.QUANTIFICATION_ERROR:
        logging.info(f'optimizing for quantification [{error.__name__}]')
        optimize_for_quantification(
            method,
            training,
            error,
            FLAGS.sample_size,
            sample_prevalences=artificial_prevalence_sampling(21*10),
            param_grid=HYPERPARAMS[learner],
            n_jobs=decide_njobs(method)
        )
    else:
        raise ValueError('unexpected value for parameter "error"')


def produce_predictions(method, test, n_prevalences=21, repeats=100):
    logging.info(f'generating predictions for test')

    learner = FLAGS.learner.lower()
    results = bert_predicion(method, test, n_prevalences, repeats) if learner == 'bert' else parallel_prediction(method, test, n_prevalences, repeats)

    true_prevalences, estim_prevalences = zip(*results)
    true_prevalences = np.asarray(true_prevalences)
    estim_prevalences = np.asarray(estim_prevalences)

    return true_prevalences, estim_prevalences


def parallel_prediction(method, test, n_prevalences, repeats):
    results = Parallel(n_jobs=decide_njobs(method))(
        delayed(test_method)(sample, method) for sample in tqdm(
            test.artificial_sampling_generator(FLAGS.sample_size, n_prevalences=n_prevalences, repeats=repeats),
            total=n_prevalences*repeats,
            desc='testing'
        )
    )
    return results


def bert_predicion(method, test, n_prevalences, repeats):
    results = [test_method(sample, method) for sample in tqdm(
            test.artificial_sampling_generator(FLAGS.sample_size, n_prevalences=n_prevalences, repeats=repeats),
            total=n_prevalences*repeats,
            desc='testing'
        )]
    return results


def test_method(sample, method):
        true_prevalence = sample.prevalence()
        estim_prevalence = method.quantify(sample.documents)
        return true_prevalence, estim_prevalence


def evaluate_experiment(true_prevalences, estim_prevalences, n_prevalences=21, repeats=100, show_plot=False):
    true_ave = true_prevalences.reshape(n_prevalences, repeats).mean(axis=1)
    estim_ave = estim_prevalences.reshape(n_prevalences, repeats).mean(axis=1)
    estim_std = estim_prevalences.reshape(n_prevalences, repeats).std(axis=1)
    print('\nTrueP->mean(Phat)(std(Phat))\n'+'='*22)
    for true, estim, std in zip(true_ave, estim_ave, estim_std):
        print(f'{true:.3f}->{estim:.3f}(+-{std:.4f})')

    print('\nEvaluation Metrics:\n'+'='*22)
    for eval in [errors.mae, errors.mrae]:
        print(f'\t{eval.__name__}={eval(true_prevalences, estim_prevalences):.3f}')
    print()

    write_csv(f'{errors.mae(true_prevalences, estim_prevalences):.3f}', f'{errors.mrae(true_prevalences, estim_prevalences):.3f}')

    if show_plot:
        plot_diagonal(true_prevalences, {f'{FLAGS.method}-{FLAGS.learner}-{FLAGS.error}': estim_prevalences})

def write_csv(mae, mrae):
    import csv
    with open('results.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([FLAGS.dataset, FLAGS.method.lower(), mae, mrae])