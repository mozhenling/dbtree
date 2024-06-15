# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 16:32:20 2021

@author: mozhenling
"""
import os
import time
import json
import numpy as np

#%%--------------------- Useful tools
def get_time():
    """
    get the current time in format of YY MM DD HH SS
    """
    st = time.strftime('%Y%m%d%H%M%S', time.localtime())
    return st

class NumpyEncoder(json.JSONEncoder):
    """
     The json.dump() function is designed to work with native Python data types,
     and NumPy's ndarray is not one of them. To resolve this issue, we need
     to convert the ndarray to a Python list or another JSON-serializable
     data type before saving it to a JSON file.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_dict(dictObj, saveName):
    """
    save a dictionary object
    """
    if not saveName.endswith('.json'):
        saveName = os.path.splitext(saveName)[0] + '.json'
        print("The save format has been changed to .json in the latest update")
    # Save the dictionary to a JSON file with UTF-8 encoding
    with open(saveName, 'w', encoding='utf-8') as f:
        json.dump(dictObj, f, cls=NumpyEncoder)

def read_dict(readName):
    """
    read teh dictionary object
    """
    try:
        # Load the dictionary from the JSON file with UTF-8 encoding
        with open(readName, 'r', encoding='utf-8') as f:
            dictObj = json.load(f)
    except:
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