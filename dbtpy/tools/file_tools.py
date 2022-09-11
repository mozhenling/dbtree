# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 16:32:20 2021

@author: mozhenling
"""
import os
import time

#%%--------------------- Useful tools
def get_time():
    """
    get the current time in format of YY MM DD HH SS
    """
    st = time.strftime('%Y%m%d%H%M%S', time.localtime())
    return st

def save_dict(dictObj, saveName):
    """
    save a dictionary object into text 
    """
    f = open(saveName, 'w')
    f.write(str(dictObj))
    f.close()

def read_dict(readName):
    """
    read teh dictionary object
    """
    f = open(readName, 'r')
    a = f.read()
    dictObj = eval(a)
    f.close()
    return dictObj

def file_type(path):
    """
    Split the pathname path into a pair (root, ext) such that root + ext == path
    """
    root, ext=os.path.splitext(path)
    return root, ext
def path_join(str1, str2):
    """
    join the str1 and str2 as a path
    """
    return os.path.join(str1, str2)
#%%