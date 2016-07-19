#!/usr/bin/env python
#-*-coding:utf-8-*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import settings

source_data=[]
target_data=[]
test_data = []
dictionary={}
inv_dictionary={}
vocab_size = 0
data_set = [ [] for bucket in settings.buckets ]
train_set = [ [] for bucket in settings.buckets ]
dev_set = [ [] for bucket in settings.buckets ]



def load_data():
    global source_data,target_data,dictionary,inv_dictionary,vocab_size,test_data
    #source
    lines = open(settings.source_wid_file).read().rstrip().split('\n')
    source_data = [map(int,line.split()) for line in lines]
    #target
    lines = open(settings.target_wid_file).read().rstrip().split('\n')
    target_data = [map(int,line.split()) for line in lines]
    #test source
    lines = open(settings.test_source_text_file).read().decode('utf-8').rstrip().split('\n')
    test_data = [line.split() for line in lines]
    #dictionary
    lines = open(settings.vocab_file).read().decode('utf-8').rstrip().split('\n')
    dictionary = {line.split('\t')[0]:int(line.split('\t')[1]) for line in lines}
    #inv dictionary
    lines = open(settings.inv_vocab_file).read().decode('utf-8').rstrip().split('\n')
    inv_dictionary = {int(line.split('\t')[0]):line.split('\t')[1] for line in lines}
    vocab_size = len(dictionary)

def prepare_dataset():
    global data_set
    data_set = [ [] for bucket in settings.buckets ]
    for i in xrange(len(source_data)):
        source_wids, target_wids, bucket_id = format_source_target(source_data[i],target_data[i])
        if bucket_id is not  None:
            data_set[bucket_id].append((source_wids,target_wids))


def format_source_target(source_wids,target_wids,test=False):
    if not test:
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
            return (source_wids,target_wids,bucket_id)
    return (None,None,None)

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

def token_ids_to_wordlist(token_ids):
    return [inv_dictionary[wid] for wid in token_ids]


if __name__ == '__main__':
    load_data()
    prepare_dataset()
    print test_data[0]
    #load_and_prepare_data()
    #print [len(one) for one in data_set], ' data in each bucket'
