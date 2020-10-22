from .aggregative import *
from .non_aggregative import *

AGGREGATIVE_METHODS = {
    ClassifyAndCount,
    AdjustedClassifyAndCount,
    ProbabilisticClassifyAndCount,
    ProbabilisticAdjustedClassifyAndCount,
    ExplicitLossMinimisation,
    ExpectationMaximizationQuantifier,
    HellingerDistanceY,
    QuaNet
}

NON_AGGREGATIVE_METHODS = {
    MaximumLikelihoodPrevalenceEstimation
}

QUANTIFICATION_METHODS = AGGREGATIVE_METHODS | NON_AGGREGATIVE_METHODS

NEURAL_METHODS = {
    QuaNet
}

# common alisases
CC = ClassifyAndCount
ACC = AdjustedClassifyAndCount
PCC = ProbabilisticClassifyAndCount
PACC = ProbabilisticAdjustedClassifyAndCount
ELM = ExplicitLossMinimisation
EMQ = ExpectationMaximizationQuantifier
HDy = HellingerDistanceY
MLPE = MaximumLikelihoodPrevalenceEstimation


