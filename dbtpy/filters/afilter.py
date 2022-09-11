# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 14:54:05 2022

@author: mozhenling
"""

from dbtpy.filters.svmdpy import svmd
from dbtpy.filters.vmdpy  import vmd
from dbtpy.filters.meyerpy  import meyer

filters_dict = {}
filters_dict['svmd'] = svmd
filters_dict['vmd'] = vmd
filters_dict['meyer'] = meyer

def filter_fun(sig_real, filterBase='meyer', **kwargs):
    """
    -sig_real: the real valued signal
    -filterBase: the base filter
    """
    return filters_dict[filterBase](sig_real,  **kwargs)