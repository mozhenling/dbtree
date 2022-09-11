# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 10:55:36 2022

@author: mozhenling
"""
import numpy as np
from scipy.linalg import norm
################################################################################
# ------------------------------------------------------------------------------
# --------------the Hoyer index, see ref. [4],[5]
# ------------------------------------------------------------------------------
def hoyer(seq):
    """
    the Hoyer index of a sequence

    input:
        seq # 1-d arrary_like sequence

    ref.:
        Hoyer, Patrik O. "Non-negative matrix factorization with sparseness constraints.
        " Journal of machine learning research 5.9 (2004).
    """
    N = len(seq)
    return (np.sqrt(N) - norm(seq, 1) / norm(seq, 2)) / (np.sqrt(N) - 1)