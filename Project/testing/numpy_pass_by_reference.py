import numpy as np

def fun1(arr):
   return arr[3:]

def fun2(arr):
   arr = arr[3:]
   
def fun3(arr):
    np.delete(arr,[0,1,2],axis=0)
    
a = np.array([3, 4, 5,6,7,8,7,8,7,7,7,7])

fun3(a)
print(a) # prints [3, 4, 5]

fun2(a)
print(a) # prints [3, 4, 5]

a=fun1(a)
print(a) # prints [0, 1, 2]# -*- coding: utf-8 -*-

