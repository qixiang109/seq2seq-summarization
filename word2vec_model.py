#!/usr/bin/env python
#-*-coding:utf-8-*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import gensim
import data_utils
import settings
import numpy as np


def run(corpus,size):
    ret = np.random.uniform(-0.1,0.1,size=(len(data_utils.inv_dictionary),settings.size))
    model = gensim.models.Word2Vec(corpus, size ,workers=10)
    for word in model.vocab:
        for k in xrange(size):
            ret[data_utils.dictionary[word],k] = model[word][k]
    return ret
