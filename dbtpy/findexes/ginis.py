# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 10:52:58 2022

@author: mozhenling
"""
import numpy as np
################################################################################
# ------------------------------------------------------------------------------
# --------------the Gini index, see ref. [1],[2]
# see also https://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
# ------------------------------------------------------------------------------
def gini(seq):
    """
    the Gini index of a sequence

    inputs:
        -seq # 1-d sequence

    ref.:
        Wang, Dong. "Some further thoughts about spectral kurtosis, spectral L2/L1 norm,
        spectral smoothness index and spectral Gini index for characterizing repetitive
        transients." Mechanical Systems and Signal Processing 108 (2018): 360-368.
    """
    minValue = min(seq)
    if minValue <= 0:
        seq -= minValue  # non-negtives
        seq += np.spacing(1)  # non-zeros
    seq_r = np.sort(seq)  # sort in ascending order
    N = len(seq)
    # python start with 0 for i
    numerator = sum([(2 * (i + 1) - N - 1) * seq_r_i for i, seq_r_i in enumerate(seq_r)])
    denominator = N * sum(seq)  # norm(se, 1)
    return numerator / denominator
# ------------------------------------------------------------------------------
def gini2(seq):
    """
    ref.:
        B. Hou, D. Wang, T. Yan, Y. Wang, Z. Peng, and K.-L. Tsui, “Gini Indices II
        and III: Two New Sparsity Measures and Their Applications to Machine Condition
        Monitoring,” IEEE/ASME Trans. Mechatronics, pp. 1–1, 2021, doi: 10.1109/TMECH.2021.3100532.
    """
    minValue = min(seq)
    if minValue <= 0:
        seq -= minValue  # non-negtives
        seq += np.spacing(1)  # non-zeros
    seq_r = np.sort(seq)  # sort in ascending order
    N = len(seq)
    # python start with 0 for i
    M_G1 = sum([(2 * N - 2 * n + 1) * Cn / N ** 2 for n, Cn in enumerate(seq_r)] )
    M_G2 = sum([(2 * n - 1) * Cn / N ** 2 for n, Cn in enumerate(seq_r)])

    return 1 - M_G1 / M_G2
# ------------------------------------------------------------------------------
def gini3(seq):
    """
    ref.:
        B. Hou, D. Wang, T. Yan, Y. Wang, Z. Peng, and K.-L. Tsui, “Gini Indices II
        and III: Two New Sparsity Measures and Their Applications to Machine Condition
        Monitoring,” IEEE/ASME Trans. Mechatronics, pp. 1–1, 2021, doi: 10.1109/TMECH.2021.3100532.
    """
    minValue = min(seq)
    if minValue <= 0:
        seq -= minValue  # non-negtives
        seq += np.spacing(1)  # non-zeros
    seq_r = np.sort(seq)  # sort in ascending order
    N = len(seq)
    M_x1 = np.mean(seq_r)
    M_G2 = sum([(2 * n - 1) * Cn / N ** 2 for n, Cn in enumerate(seq_r) ])

    return 1 - M_x1 / M_G2