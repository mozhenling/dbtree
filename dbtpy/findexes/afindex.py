# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 11:34:46 2022

@author: mozhenling
"""
from dbtpy.findexes.sigto import sig_real_to_env, sig_real_to_se, sig_real_to_ses, sig_real_to_sses
from scipy.stats import kurtosis
from dbtpy.findexes.entropies import entropy, pentropy, negentropy
from dbtpy.findexes.ginis import  gini,  gini2, gini3
from dbtpy.findexes.harmonics import harEstimation, vanillaSNR, harL2L1norm, CHNR, harkurtosis
from dbtpy.findexes.hoyers import  hoyer
from dbtpy.findexes.pq_norms import pq_norm
from dbtpy.findexes.rsmooths import rsmooth

###############################################################################
#------------------------------------------------------------------------------
findexes_dict = {}
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# --------------fault index based on the kurtosis
findexes_dict['kurt'] = kurtosis
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# --------------fault index based on the entropy
findexes_dict['entropy'] = entropy
findexes_dict['pentropy'] = pentropy
findexes_dict['negentropy'] = negentropy
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# --------------fault index based on the gini index
findexes_dict['gini'] = gini
findexes_dict['gini1'] = gini
findexes_dict['gini2'] = gini2
findexes_dict['gini3'] = gini3
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -------------- fault index related to the fault harmonics
# findexes_dict['harEst'] = harEstimation 
findexes_dict['vSNR'] = vanillaSNR
findexes_dict['harL2L1'] = harL2L1norm
findexes_dict['harkurt'] =harkurtosis
findexes_dict['CHNR'] = CHNR
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# --------------fault index based on the  hoyer measure
findexes_dict['hoyer'] = hoyer
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# --------------fault index based on the pq_norm
findexes_dict['pq_norm'] = pq_norm
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#--------------fault index based on the reciprocal of the smoothness index 
findexes_dict['rsmooth'] = rsmooth

###############################################################################
# -----------------------------------------------------------------------------
# -------------- fault index based on different signal domains
# -----------------------------------------------------------------------------
def findex_fun(sig_real,  findexBase = 'kurt', sigD = None, gain = 1, offset = 0, **kwargs):
    """
    -sig_real: the real valued signal
    -findexBase: the base fault index
    -sigD: the signal domain for calculating the fault index
    -gain: if sign = 1, keep the orignal sign. Otherwise, take the negative version by gain = -1
           It can be also used as a normalizing constant
    -offset: offset the fault index
    """
    if  sigD is None or sigD =='real' :
        out =  findexes_dict[findexBase](sig_real, **kwargs)
    elif sigD =='env':
        out = findexes_dict[findexBase](sig_real_to_env(sig_real), **kwargs)
    elif sigD =='se':
        out =  findexes_dict[findexBase](sig_real_to_se(sig_real), **kwargs)
    elif sigD == 'ses':
        out =  findexes_dict[findexBase](sig_real_to_ses(sig_real), **kwargs)
    elif sigD == 'sses':
        out =  findexes_dict[findexBase](sig_real_to_sses(sig_real), **kwargs)
    elif sigD == 'se_ses':
        out =  ( findexes_dict[findexBase](sig_real_to_se(sig_real), **kwargs) +
                 findexes_dict[findexBase](sig_real_to_ses(sig_real), **kwargs)) / 2
    else:
        raise NotImplementedError
        
    return  gain*out - offset