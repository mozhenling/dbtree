# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 10:57:16 2022

@author: mozhenling
"""
import numpy as np
################################################################################
# ------------------------------------------------------------------------------
# --------------the entropy, see ref. [6]
# ------------------------------------------------------------------------------
# def avoid_inf(seq):
#     return seq[seq!=0]
avoid_inf = np.spacing(1) # prevent troubles caused by zeros    
def entropy(p):
    """
    the Shannon entropy
    input:
        -p  # 1-d arrary_like of probabilities
    ref.:
        Li, Yongbo, et al. "The entropy algorithm and its variants in the fault
        diagnosis of rotating machinery: A review." Ieee Access 6 (2018): 66723-66741.
    """
    return -sum([pxi * np.log2(pxi + avoid_inf) for pxi in p])

def pentropy(sub_sig):
    """
    the energy/power shannon entropy

    input:
        -sub_sig # list of subsignals e.g. sub_sig = [[sub_sig1], [sub_sig2], ...]

    ref.:
        Su, Houjun, et al. "New method of fault diagnosis of rotating machinery
        based on distance of information entropy." Frontiers of Mechanical Engineering
        6.2 (2011): 249.
    """
    e_list = [sum(sub_s ** 2) for sub_s in sub_sig]
    e_sum = sum(e_list) + avoid_inf
    p = [e / e_sum for e in e_list]

    return entropy(p)

################################################################################
def negentropy(seq):
    """
    the negetropy of a sequence
    ref:
        Antoni, Jerome. "The infogram: Entropic evidence of the signature of
            repetitive transients." Mechanical Systems and Signal Processing 74 
            (2016): 73-94.
    """
    seq += avoid_inf
    return np.mean( (seq / np.mean(seq) ) * np.log( seq / np.mean(seq)) )