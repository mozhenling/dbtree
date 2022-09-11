# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 10:51:28 2022

@author: mozhenling
"""
import numpy as np
from scipy.stats.mstats import gmean
# ------------------------------------------------------------------------------
# --------------the smoothness index, see ref. [1],[2]
# ------------------------------------------------------------------------------
def rsmooth(seq):
    """
    the reciprocal of the smoothness index of a sequence

    inputs:
        -seq # 1-d sequence

    ref.:
        Wang, Dong. "Some further thoughts about spectral kurtosis, spectral L2/L1 norm,
        spectral smoothness index and spectral Gini index for characterizing repetitive
        transients." Mechanical Systems and Signal Processing 108 (2018): 360-368.
        
    """
    seq_non_zeros = seq[seq!=0]
    return np.mean(seq_non_zeros) / gmean(seq_non_zeros)