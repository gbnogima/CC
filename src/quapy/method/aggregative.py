from .base import *
from sklearn.calibration import CalibratedClassifierCV
from quapy.classification.svmperf import SVMperf
from quapy.dataset.text import LabelledCollection
from quapy.functional import *


# Abstract classes
# ------------------------------------
class AggregativeQuantifier(BaseQuantifier):

    @abstractmethod
    def fit(self, data: LabelledCollection, fit_learner=True, *args): ...

    def classify(self, documents):
        return self.learner.predict(documents)

    def get_params(self):
        return self.learner.get_params()

    def set_params(self, **parameters):
        self.learner.set_params(**parameters)


class AggregativeProbabilisticQuantifier(AggregativeQuantifier):

    def soft_classify(self, data):
        return self.learner.predict_proba(data)[:, self.learner.classes_ == 1].flatten()


# Helper
# ------------------------------------
def training_helper(learner,
                    data: LabelledCollection,
                    fit_learner: bool = True,
                    ensure_probabilistic=False,
                    train_val_split=None):
    if fit_learner:
        if ensure_probabilistic:
            if not hasattr(learner, 'predict_proba'):
                print(f'The learner {learner.__class__.__name__} does not seem to be probabilistic. '
                      f'The learner will be calibrated.')
                learner = CalibratedClassifierCV(learner, cv=5)
        if train_val_split is not None:
            if not (0 < train_val_split < 1):
                raise ValueError(f'train/val split {train_val_split} out of range, must be in (0,1)')
            train, unused = data.split_stratified(train_size=train_val_split)
        else:
            train, unused = data, None
        learner.fit(train.documents, train.labels)
    else:
        if ensure_probabilistic:
            if not hasattr(learner, 'predict_proba'):
                raise AssertionError('error: the learner cannot be calibrated since fit_learner is set to False')
        unused = data

    return learner, unused


# Methods
# ------------------------------------
class ClassifyAndCount(AggregativeQuantifier):

    def __init__(self, learner):
        self.learner = learner

    def fit(self, data: LabelledCollection, fit_learner=True, *args):
        self.learner, _ = training_helper(self.learner, data, fit_learner)
        return self

    def quantify(self, documents, *args):
        classification = self.classify(documents)           # classify
        return prevalence_from_predictions(classification)  # & count


class AdjustedClassifyAndCount(AggregativeQuantifier):

    def __init__(self, learner):
        self.learner = learner

    def fit(self, data: LabelledCollection, fit_learner=True, train_val_split=0.6):
        self.learner, validation = training_helper(self.learner, data, fit_learner, train_val_split=train_val_split)
        self.cc = ClassifyAndCount(self.learner)
        self.tpr_, self.fpr_ = classifier_tpr_fpr(self.classify, validation)
        return self

    def quantify(self, documents, *args):
        cc = self.cc.quantify(documents)
        acc = adjusted_quantification(cc, self.tpr_, self.fpr_)
        return acc

    def classify(self, data):
        return self.cc.classify(data)


class ProbabilisticClassifyAndCount(AggregativeProbabilisticQuantifier):
    def __init__(self, learner):
        self.learner = learner

    def fit(self, data : LabelledCollection, fit_learner=True, *args):
        self.learner, _ = training_helper(self.learner, data, fit_learner, ensure_probabilistic=True)
        return self

    def quantify(self, documents, *args):
        posteriors = self.soft_classify(documents)                        # classify
        return prevalence_from_probabilities(posteriors, binarize=False)  # & count


class ProbabilisticAdjustedClassifyAndCount(AggregativeQuantifier):

    def __init__(self, learner):
        self.learner = learner

    def fit(self, data: LabelledCollection, fit_learner=True, train_val_split=0.6):
        learner, validation = training_helper(
            self.learner, data, fit_learner, ensure_probabilistic=True, train_val_split=train_val_split
        )
        self.pcc = ProbabilisticClassifyAndCount(learner)
        self.tpr_, self.fpr_ = classifier_tpr_fpr(self.pcc.soft_classify, validation)
        return self

    def quantify(self, documents, *args):
        pcc = self.pcc.quantify(documents)
        pacc = adjusted_quantification(pcc, self.tpr_, self.fpr_)
        return pacc

    def classify(self, data):
        return self.pcc.classify(data)


class ExpectationMaximizationQuantifier(AggregativeProbabilisticQuantifier):

    MAX_ITER = 1000

    def __init__(self, learner, verbose=False):
        self.learner = learner
        self.verbose = verbose

    def fit(self, data: LabelledCollection, fit_learner=True, *args):
        self.learner, _ = training_helper(self.learner, data, fit_learner, ensure_probabilistic=True)
        self.train_prevalence = prevalence_from_predictions(data.labels)
        return self

    def quantify(self, X, y=None, epsilon=1e-4):
        tr_prev=self.train_prevalence
        posteriors = self.soft_classify(X)
        return self.EM(tr_prev, posteriors, self.verbose, y, epsilon)

    @classmethod
    def EM(cls, tr_prev, posterior_probabilities, verbose=False, true_labels=None, epsilon=1e-4):
        Px = posterior_probabilities
        Px_pos = Px
        Px_neg = 1. - Px
        trueprev = prevalence_from_predictions(true_labels) if true_labels is not None else -1

        Ptr_pos = tr_prev  #Ptr(y=+1)
        Ptr_neg = 1-Ptr_pos  #Ptr(y=0)
        qs_pos, qs_neg = Ptr_pos, Ptr_neg       # i.e., prevalence(ytr)

        s, converged = 0, False
        qs_pos_prev_ = None
        while not converged and s < ExpectationMaximizationQuantifier.MAX_ITER:
            # E-step: ps is Ps(y=+1|xi)
            pos_factor = (qs_pos / Ptr_pos) * Px_pos
            neg_factor = (qs_neg / Ptr_neg) * Px_neg
            ps = pos_factor / (pos_factor + neg_factor)

            # M-step: qs_pos is Ps+1(y=+1)
            qs_pos = np.mean(ps)
            qs_neg = 1 - qs_pos

            if verbose:
                print(('s={} qs_pos={:.6f}'+('' if y is None else ' true={:.6f}'.format(trueprev))).format(s,qs_pos))

            if qs_pos_prev_ is not None and abs(qs_pos - qs_pos_prev_) < epsilon and s>10:
                converged = True

            qs_pos_prev_ = qs_pos
            s += 1

        if verbose:
            print('-'*80)

        if not converged:
            raise UserWarning('the method has reached the maximum number of iterations, it might have not converged')

        return qs_pos


class ExplicitLossMinimisation(AggregativeQuantifier):

    def __init__(self, svmperf_base, loss, **kwargs):
        self.learner = SVMperf(svmperf_base, loss=loss, **kwargs)

    def fit(self, data: LabelledCollection, fit_learner=True, *args):
        assert fit_learner, 'the method requires that fit_learner=True'
        self.learner.fit(data.documents, data.labels)
        return self

    def quantify(self, X, y=None):
        predictions = self.learner.predict(X)
        return prevalence_from_predictions(predictions)


class SVMQ(ExplicitLossMinimisation):
    def __init__(self, svmperf_base, **kwargs):
        super(SVMQ, self).__init__(svmperf_base, loss='q', **kwargs)


class SVMKLD(ExplicitLossMinimisation):
    def __init__(self, svmperf_base, **kwargs):
        super(SVMKLD, self).__init__(svmperf_base, loss='kld', **kwargs)


class SVMNKLD(ExplicitLossMinimisation):
    def __init__(self, svmperf_base, **kwargs):
        super(SVMNKLD, self).__init__(svmperf_base, loss='nkld', **kwargs)


class SVMAE(ExplicitLossMinimisation):
    def __init__(self, svmperf_base, **kwargs):
        super(SVMAE, self).__init__(svmperf_base, loss='mae', **kwargs)


class SVMRAE(ExplicitLossMinimisation):
    def __init__(self, svmperf_base, **kwargs):
        super(SVMRAE, self).__init__(svmperf_base, loss='mrae', **kwargs)


class HellingerDistanceY(AggregativeProbabilisticQuantifier):
    """
    Implementation of the method based on the Hellinger Distance y (HDy) proposed by
    González-Castro, V., Alaiz-Rodrı́guez, R., and Alegre, E. (2013). Class distribution
    estimation based on the Hellinger distance. Information Sciences, 218:146–164.
    """

    def __init__(self, learner):
        self.learner = learner

    def fit(self, data: LabelledCollection, fit_learner=True, train_val_split=0.6):
        self.learner, validation = training_helper(
            self.learner, data, fit_learner, ensure_probabilistic=True, train_val_split=train_val_split)
        Px = self.soft_classify(validation.documents)
        self.Pxy1 = Px[validation.labels == 1]
        self.Pxy0 = Px[validation.labels == 0]
        return self

    def quantify(self, documents, *args):
        # "In this work, the number of bins b used in HDx and HDy was chosen from 10 to 110 in steps of 10,
        # and the final estimated a priori probability was taken as the median of these 11 estimates."
        # (González-Castro, et al., 2013).

        Px = self.soft_classify(documents)

        prev_estimations = []
        for bins in np.linspace(10, 110, 11, dtype=int): #[10, 20, 30, ..., 100, 110]
            Pxy0_density, _ = np.histogram(self.Pxy0, bins=bins, range=(0, 1), density=True)
            Pxy1_density, _ = np.histogram(self.Pxy1, bins=bins, range=(0, 1), density=True)

            Px_test, _ = np.histogram(Px, bins=bins, range=(0, 1), density=True)

            prev_selected, min_dist = None, None
            for prev in prevalence_linspace(n_prevalences=100, repeat=1, smooth_limits_epsilon=0.0):
                Px_train = prev*Pxy1_density + (1 - prev)*Pxy0_density
                hdy = HellingerDistanceY.HellingerDistance(Px_train, Px_test)
                if prev_selected is None or hdy < min_dist:
                    prev_selected, min_dist = prev, hdy
            prev_estimations.append(prev_selected)

        return np.median(prev_estimations)

    @classmethod
    def HellingerDistance(cls, P, Q):
        return np.sqrt(np.sum((np.sqrt(P) - np.sqrt(Q))**2))


from . import neural
QuaNet = neural.QuaNetTrainer
