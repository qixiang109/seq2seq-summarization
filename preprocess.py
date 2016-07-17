#-encoding=utf-8
import sys
import numpy
import re
import settings
from optparse import OptionParser

dictionary = {}
dictionary[settings.PAD] = settings.PAD_ID
dictionary[settings.GO] = settings.GO_ID
dictionary[settings.EOS] = settings.EOS_ID
dictionary[settings.UNK] = settings.UNK_ID


def read_text(filename, max_lines=None):
    docs = []
    pattern_blank = re.compile('\s+')
    for d, line in enumerate(open(filename)):
        if max_lines is not None and d >= max_lines:
            break
        wordlist = pattern_blank.subn(
            ' ', line)[0].strip().decode('utf-8').split()
        docs.append(wordlist)
    return docs


def count_vocabulary(docs, vocab_size=None):
    global dictionary
    df = {}
    for wordlist in docs:
        for word in set(wordlist):
            if not word in df:
                df[word] = 0
            df[word] += 1
    for word, df in sorted(df.items(), key=lambda d: d[1], reverse=True):
        if len(dictionary) == settings.vocab_size:
            break
        dictionary[word] = len(dictionary)


def wordlist_to_token_ids(wordlist):
    if isinstance(wordlist, str):
        wordlist = wordlist.decode('utf-8')
    if isinstance(wordlist, unicode):
        wordlist = list(wordlist)
    if not isinstance(wordlist, list):
        print 'sentence', wordlist, 'not str, unicode or list'
        return None
    ids = [-1] * len(wordlist)
    for i, word in enumerate(wordlist):
        if word in dictionary:
            ids[i] = dictionary[word]
        else:
            ids[i] = dictionary[settings.UNK]
    return ids


def docs_to_token_ids(docs):
    ret = []
    for wordlist in docs:
        ids = wordlist_to_token_ids(wordlist)
        ret.append(ids)
    return ret


def preprocess():
    source_docs = read_text(settings.source_text_file, settings.preprocess_num)
    target_docs = read_text(settings.target_text_file, settings.preprocess_num)
    count_vocabulary(source_docs + target_docs)
    source_wordids = docs_to_token_ids(source_docs)
    target_wordids = docs_to_token_ids(target_docs)
    with open(settings.source_wid_file, 'w') as fw:
        for widlist in source_wordids:
            fw.write(' '.join(map(str,widlist)) + '\n')
    with open(settings.target_wid_file, 'w') as fw:
        for widlist in target_wordids:
            fw.write(' '.join(map(str,widlist)) + '\n')
    with open(settings.vocab_file, 'w') as fw:
        for word, idx in sorted(dictionary.items(), key=lambda d:d[1], reverse=False):
            fw.write(word + '\t' + str(idx) + '\n')
    with open(settings.inv_vocab_file, 'w') as fw:
        for word, idx in sorted(dictionary.items(), key=lambda d:d[1], reverse=False):
            fw.write(str(idx) + '\t' + word + '\n')

def main():
    preprocess()

if __name__ == '__main__':
    main()
