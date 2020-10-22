import numpy as np
from sklearn.metrics import f1_score
from settings import SAMPLE_SIZE


def tpr(y_true, y_pred):
    return _safe_mean(y_pred[y_true == 1])


def fpr(y_true, y_pred):
    return _safe_mean(y_pred[y_true == 0])


def precision(y_true, y_pred):
    return _safe_mean(y_true[y_pred == 1])


def recall(y_true, y_pred):
    return tpr(y_true, y_pred)


def fomr(y_true, y_pred):
    return _safe_mean(y_true[y_pred == 0])


def _safe_mean(array):
    return (array.mean() if array.size > 0 else 0)


def f1e(y_true, y_pred):
    pos_prev = y_true.mean()
    minoritary_class = 1 if pos_prev < 0.5 else 0
    return 1. - f1_score(y_true, y_pred, pos_label=minoritary_class, average='binary')


def acce(y_true, y_pred):
    acc = (y_true == y_pred).mean()
    return 1. - acc


def mae(prevs, prevs_hat):
    return ae(prevs, prevs_hat).mean()


def mse(prevs, prevs_hat):
    return se(prevs, prevs_hat).mean()


def mkld(prevs, prevs_hat):
    return kld(prevs, prevs_hat).mean()


def mnkld(prevs, prevs_hat):
    return nkld(prevs, prevs_hat).mean()


def mrae(prevs, prevs_hat, eps=1/(2 * SAMPLE_SIZE)):
    return RAE(prevs, prevs_hat, eps).mean()


def ae(p, p_hat):
    return abs(p_hat-p)


def se(p, p_hat):
    return (p_hat-p)**2


def kld(p, p_hat, eps=1 / (2 * SAMPLE_SIZE)):
    sp = p+eps
    sp_hat = p_hat + eps
    first = sp*np.log(sp/sp_hat)
    second = (1.-sp)*np.log(abs((1.-sp)/(1.-sp_hat)))
    return first + second


def nkld(p, p_hat, eps=1 / (2 * SAMPLE_SIZE)):
    ekld = np.exp(kld(p, p_hat, eps))
    return 2. * ekld / (1 + ekld) - 1.


def bin_smooth(p, eps):
    return (eps + p) / (1. + eps * 2.)


# it was proposed in literature an eps = 1/(2*T), with T the size of the test set
def RAE(p, p_hat, eps=1./(2. * SAMPLE_SIZE)):
    return 0.5*RAEpos(p, p_hat, eps) + 0.5*RAEpos(1-p, 1-p_hat, eps)


def RAEpos(p, p_hat, eps=1./(2. * SAMPLE_SIZE)):
    p = bin_smooth(p, eps)
    p_hat = bin_smooth(p_hat, eps)
    return abs(p_hat - p) / p


CLASSIFICATION_ERROR = {f1e, acce}
QUANTIFICATION_ERROR = {mae, mse, mkld, mnkld, mrae}
