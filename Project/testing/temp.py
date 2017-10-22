# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 17:16:02 2017

@author: JanneK
"""

from multiprocessing import Pool

def f(x):
    return x*x

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    p = Pool(5)
    print(p.map(f, [1, 2, 3]))