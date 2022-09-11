# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 10:55:01 2022

@author: mozhenling
"""
from scipy.linalg import norm
from scipy.stats.mstats import gmean
################################################################################
# ------------------------------------------------------------------------------
# --------------the Lp/Lq norm without normalization, see ref. [1],[2]
# ------------------------------------------------------------------------------
def pq_norm(seq, p, q):
    """
    the Lp/Lq norm of a sequence

    inputs:
        -seq # 1-d sequence
        -p   # Lp norm
        -q   # Lq norm

    ref.:
        Wang, Dong. "Some further thoughts about spectral kurtosis, spectral L2/L1 norm,
        spectral smoothness index and spectral Gini index for characterizing repetitive
        transients." Mechanical Systems and Signal Processing 108 (2018): 360-368.
    """
    N = len(seq)
    if q == 0:
        return (N ** (-1 / p)) * norm(seq, p) / gmean(seq)
    else:
        return (N ** (1 / q - 1 / p)) * norm(seq, p) / norm(seq, q)