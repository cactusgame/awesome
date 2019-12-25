import os
import imp
import math
from src.base.config import cfg


def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def get_model_timestamp():
    """
    get the TF model timestamp. The timestamp only exist after the model has been generated
    :return:
    """
    # get model timestamp
    for root, dirs, files in os.walk(cfg.TARGET_DIR, topdown=False):
        if root == cfg.TARGET_MODEL_DIR:
            return dirs[0]
    return None


def import_from_uri(uri, absl=True):
    """
    import a module according to given path
    :param uri:
    :param absl:
    :return:
    """
    if not absl:
        uri = os.path.normpath(os.path.join(os.path.dirname(__file__), uri))
    path, fname = os.path.split(uri)
    mname, ext = os.path.splitext(fname)

    no_ext = os.path.join(path, mname)

    if os.path.exists(no_ext + '.py'):
        return imp.load_source(mname, no_ext + '.py')
