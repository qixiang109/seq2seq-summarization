#!/usr/bin/env python
#-*-coding:utf-8-*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import settings

source_data=[]
target_data=[]
dictionary={}
inv_dictionary={}
vocab_size = 0
data_set = [ [] for bucket in settings.buckets ]
train_set = [ [] for bucket in settings.buckets ]
dev_set = [ [] for bucket in settings.buckets ]

def load_data():
    global source_data,target_data,dictionary,inv_dictionary,vocab_size
    #source
    lines = open(settings.source_wid_file).read().rstrip().split('\n')
    source_data = [map(int,line.split()) for line in lines]
    #target
    lines = open(settings.target_wid_file).read().rstrip().split('\n')
    target_data = [map(int,line.split()) for line in lines]
    #dictionary
    lines = open(settings.vocab_file).read().decode('utf-8').rstrip().split('\n')
    dictionary = {line.split('\t')[0]:int(line.split('\t')[1]) for line in lines}
    #inv dictionary
    lines = open(settings.inv_vocab_file).read().decode('utf-8').rstrip().split('\n')
    inv_dictionary = {int(line.split('\t')[0]):line.split('\t')[1] for line in lines}
    vocab_size = len(dictionary)

def prepare_dataset():
    global data_set
    #add EOS to target
    for i, target_data_i in enumerate(target_data):
        target_data[i] = target_data_i + [settings.EOS_ID]
    #add GO to target
    if settings.add_go_to_target:
        for i, target_data_i in enumerate(target_data):
            target_data[i] = [settings.GO_ID] + target_data_i
    #reverse source
    if settings.reverse_source:
        for i,source_data_i in enumerate(source_data):
            source_data_i.reverse()
            source_data[i] = source_data_i
    #pad to buckets
    num_data = len(source_data)
    for i in xrange(num_data):
        source_data_i = source_data[i]
        target_data_i = target_data[i]
        len_source_data_i = len(source_data_i)
        len_target_data_i = len(target_data_i)
        for l, (source_bucket_length, target_bucket_length) in enumerate(settings.buckets):
            if len_source_data_i <= source_bucket_length and len_target_data_i <= target_bucket_length:
                if settings.reverse_source:
                    source_data_i = [settings.PAD_ID]*(source_bucket_length-len_source_data_i)+source_data_i
                else:
                    source_data_i = source_data_i + [settings.PAD_ID]*(source_bucket_length-len_source_data_i)
                target_data_i = target_data_i + [settings.PAD_ID]*(target_bucket_length-len_target_data_i)
                data_set[l].append((source_data_i,target_data_i))
                break


def format_source_target(source_wids,target_wids):
    target_wids = target_wids+[settings.EOS_ID]
    if settings.add_go_to_target:
        target_wids = [settings.GO_ID] + target_wids
    if settings.reverse_source:
        source_wids = list(reversed(source_wids))
    lsource = len(source_wids)
    ltarget = len(target_wids)
    for bucket_id, (source_bucket_length, target_bucket_length) in enumerate(settings.buckets):
        if lsource <= source_bucket_length and ltarget <= target_bucket_length:
            target_wids = target_wids + [settings.PAD_ID]*(target_bucket_length-ltarget)
            if settings.reverse_source:
                source_wids = [settings.PAD_ID]*(source_bucket_length-lsource)+source_wids
            else:
                source_wids = source_wids + [settings.PAD_ID]*(source_bucket_length-lsource)
            break
    return (source_wids,target_wids,bucket_id)

def split_train_dev():
    global train_set,dev_set
    for i, one_set in enumerate(data_set):
        spliter = int((1-settings.dev_ratio) * len(one_set))
        train_set[i] = data_set[i][:spliter]
        dev_set[i] = data_set[i][spliter:]

def load_and_prepare_data():
    load_data()
    prepare_dataset()
    split_train_dev()

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

if __name__ == '__main__':
    load_and_prepare_data()
    #print [len(one) for one in data_set], ' data in each bucket'
