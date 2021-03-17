from absl import app
from app_helper import *

flags.DEFINE_string('dataset', None, 'the path of the directory containing the dataset (train.txt and test.txt)')
flags.DEFINE_string('method', None, 'a quantificaton method (cc, acc, pcc, pacc, emq, svmq, svmkld, svmnkld, svmae, '
                                    'svmrae, hdy, mlpe, quanet)')
flags.DEFINE_string('learner', None, f'a classification learner method (svm lr mnb rf cnn)')
flags.DEFINE_integer('sample_size', settings.SAMPLE_SIZE, f'sampling size')
flags.DEFINE_string('error', 'mae', 'error to optimize for in model selection (none acce f1e mae mrae)')
flags.DEFINE_string('results_path', '../results', 'where to pickle the results as a pickle containing the true '
                                                  'prevalences and the estimated prevalences')
flags.DEFINE_string('suffix', '', 'a suffix to add to the result file path, e.g., "run0"')
flags.DEFINE_bool('plot', False, 'whether or not to plot the estimated predictions against the true predictions')
flags.mark_flags_as_required(['dataset', 'method', 'learner'])

FLAGS = flags.FLAGS


def main(_):

    benchmark = load_dataset()
    benchmark = prepare_dataset(benchmark)
    
    learner = instantiate_learner()
    method = instantiate_quantifier(learner)   

    model_selection(method, benchmark)

    true_prevalences, estim_prevalences = produce_predictions(method, benchmark.test)
    evaluate_experiment(true_prevalences, estim_prevalences, show_plot=FLAGS.plot)
    save_results(true_prevalences, estim_prevalences)


if __name__ == '__main__':
    app.run(main)
