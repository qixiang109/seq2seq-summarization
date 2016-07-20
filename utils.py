#!/usr/bin/env python
#-*-coding:utf-8-*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import random
import numpy as np

def weighted_choice(choices):
    total = sum(w for c, w in choices)
    r = random.uniform(0, total)
    upto = 0
    for c, w in choices:
        if upto + w >= r:
            return c
        upto += w
    assert False, "Shouldn't get here"

def shuffle_divide(L,m):
    v=range(0,len(L))
    np.random.shuffle(v)
    ret=[]
    i=0
    while True:
        if i+m>len(v):
            break
        ret.append([L[j] for j in v[i:(i+m)]])
        i+=m
    return ret



if __name__ == '__main__':
    weighted_choice([(0,2),(1,3),(2,1)])
    shuffle_divide([1,2,3,4,5],2)
