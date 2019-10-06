import numpy as np
from lsst.sims.featureScheduler.utils import standard_goals


def wfd_only_fp(nside=32):
    sg = standard_goals()
    not_wfd = np.where(sg['r'] != 1.)
    for key in sg:
        sg[key][not_wfd] = 0
    return sg

