#-encoding=utf-8
import sys
import numpy
import re
import settings
from optparse import OptionParser

def read_text(filename):
    source_texts=[]
    target_texts=[]
    pattern_blank = re.compile('\s+')
    for d, line in enumerate(open(filename)):
        source_text, target_text = line.rstrip().decode('utf-8').split('\t')
        source_wordlist = pattern_blank.subn(' ', source_text)[0].split()
        target_wordlist = pattern_blank.subn(' ', target_text)[0].split()
        source_texts.append(source_wordlist)
        target_texts.append(target_wordlist)
    return (source_texts,target_texts)

def count_vocabulary(texts, vocab_size=None):
    dictionary={}
    dictionary[settings.PAD] = settings.PAD_ID
    dictionary[settings.GO] = settings.GO_ID
    dictionary[settings.EOS] = settings.EOS_ID
    dictionary[settings.UNK] = settings.UNK_ID
    df = {}
    for wordlist in texts:
        for word in set(wordlist):
            if not word in df:
                df[word] = 0
            df[word] += 1
    for word, df in sorted(df.items(), key=lambda d: d[1], reverse=True):
        if len(dictionary) == settings.vocab_size:
            break
        dictionary[word] = len(dictionary)
    return dictionary


def preprocess():

    #读取训练数据，统计词典，数值化文档
    train_source_texts,train_target_texts  =  read_text(settings.train_text_file)
    dictionary = count_vocabulary(train_source_texts+train_target_texts)
    train_source_wids=[]
    train_target_wids=[]
    for wordlist in train_source_texts:
        train_source_wids.append([dictionary[word] if word in dictionary else settings.UNK_ID for word in wordlist])
    for wordlist in train_target_texts:
        train_target_wids.append([dictionary[word] if word in dictionary else settings.UNK_ID for word in wordlist])
    with open(settings.train_wid_file, 'w') as fw:
        for i in xrange(len(train_source_wids)):
            source = train_source_wids[i]
            target = train_target_wids[i]
            fw.write(' '.join(map(str,source)) +'\t'+' '.join(map(str,target))+'\n')
    with open(settings.vocab_file, 'w') as fw:
        for word, idx in sorted(dictionary.items(), key=lambda d:d[1], reverse=False):
            fw.write(word + '\t' + str(idx) + '\n')

    #读取测试数据，数值化文档
    test_source_texts,test_target_texts  =  read_text(settings.test_text_file)
    test_source_wids=[]
    test_target_wids=[]
    for wordlist in test_source_texts:
        test_source_wids.append([dictionary[word] if word in dictionary else settings.UNK_ID for word in wordlist])
    for wordlist in test_target_texts:
        test_target_wids.append([dictionary[word] if word in dictionary else settings.UNK_ID for word in wordlist])
    with open(settings.test_wid_file, 'w') as fw:
        for i in xrange(len(test_source_wids)):
            source = test_source_wids[i]
            target = test_target_wids[i]
            fw.write(' '.join(map(str,source)) +'\t'+' '.join(map(str,target))+'\n')

def main():
    preprocess()

if __name__ == '__main__':
    main()
