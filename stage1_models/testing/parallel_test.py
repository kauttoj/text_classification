# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 14:13:25 2017

@author: Jannek
"""

from functools import partial
import numpy as np
import time

import multiprocessing

def f(a, b, c):
    res=1
    for i in range(0,10000):
        res+=2*i
    return a*b*c+res/100

def myfun():
    iterable = np.random.randn(500)
    pool = multiprocessing.Pool(3)
    a=10
    b=20
    func = partial(f, a, b)
    result = pool.map(func, iterable)
    pool.close()
    pool.join()
    #print(result[0:10])
    return result

#if __name__ == "__main__":
#    main()