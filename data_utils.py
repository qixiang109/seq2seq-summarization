#-encoding=utf-8
import sys
import os
import numpy as np
import re
import settings
from tensorflow.python.platform import gfile


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    return sentence.strip().split()


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size):
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" %
              (vocabulary_path, data_path))
        vocab = {}
        with gfile.GFile(data_path, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("  processing line %d" % counter)
                tokens = basic_tokenizer(line)
                for word in tokens:
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = [settings.PAD, settings.GO, settings.EOS, settings.UNK] + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + b"\n")

def initialize_vocabulary(vocabulary_path):
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def sentence_to_token_ids(sentence, vocabulary):
    words = basic_tokenizer(sentence)
    return [vocabulary.get(w, settings.UNK_ID) for w in words]

def data_to_token_ids(data_path, target_path, vocabulary_path):
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocab)
                    tokens_file.write(
                        " ".join([str(tok) for tok in token_ids]) + "\n")

def prepare_data(data_dir):

    # Get wmt data to the specified directory.
    train_path = data_dir+'train'
    dev_path = data_dir+'dev'

    # Create vocabularies of the appropriate sizes.
    sr_vocab_path = os.path.join(data_dir, "vocab%d.sr" % settings.sr_vocab_size)
    tg_vocab_path = os.path.join(data_dir, "vocab%d.tg" % settings.tg_vocab_size)

    create_vocabulary(sr_vocab_path, train_path + ".sr",
                      settings.sr_vocab_size)
    create_vocabulary(tg_vocab_path, train_path + ".tg",
                      settings.tg_vocab_size)

    # Create token ids for the training data.
    sr_train_ids_path = train_path + (".ids%d.sr" % settings.sr_vocab_size)
    tg_train_ids_path = train_path + (".ids%d.tg" % settings.tg_vocab_size)

    data_to_token_ids(train_path + ".sr", sr_train_ids_path,
                      sr_vocab_path)
    data_to_token_ids(train_path + ".tg", tg_train_ids_path,
                      tg_vocab_path)


    # Create token ids for the development data.
    sr_dev_ids_path = dev_path + (".ids%d.sr" % settings.sr_vocab_size)
    tg_dev_ids_path = dev_path + (".ids%d.tg" % settings.tg_vocab_size)

    data_to_token_ids(dev_path + ".sr", sr_dev_ids_path,
                      sr_vocab_path)
    data_to_token_ids(dev_path + ".tg", tg_dev_ids_path,
                      tg_vocab_path)

    return (sr_train_ids_path, tg_train_ids_path,
            sr_dev_ids_path, tg_dev_ids_path,
            sr_vocab_path, tg_vocab_path)

def read_data(source_path,target_path,max_size=None):
    """
        读取ids形式的source，target数据，并做bucket操作
        输出：
            一个list，每个元素是一个bucket，bucket是（source, target）对的list,其中的souece, target 都做过预处理了
    """
    data_set = [[] for _ in settings.buckets]
    with gfile.GFile(source_path, mode="r") as source_file:
        with gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                #target 加 go, eos
                target_ids = [settings.GO_ID] + target_ids + [settings.EOS_ID]
                #bucket
                bucket_id = -1
                for i, (source_size, target_size) in enumerate(settings.buckets):
                    if len(source_ids) <= source_size and len(target_ids) <= target_size:
                        bucket_id = i
                        break
                source_bucket_size, target_bucket_size = settings.buckets[bucket_id]
                #pad
                source_ids = source_ids+[settings.PAD_ID]*(source_bucket_size-len(source_ids))
                target_ids = target_ids+[settings.PAD_ID]*(target_bucket_size-len(target_ids))
                #reverse
                if settings.reverse_source:
                    source_ids = list(reversed(source_ids))
                data_set[bucket_id].append((source_ids,target_ids))
                source, target = source_file.readline(), target_file.readline()
    return data_set

def batchize(data_set,reperm=True):
    """ 把bucket内部的数据打乱顺序之后做batch分割
        然后再打乱batch顺序
        目的是每个epoch内部batch顺序和batch内部构成都不同
        输出：
            一个list，每个元素是一个batch
            一个list，每个元素是对应batch的bucket_id
    """
    batch_data_sets=[]
    batch_bucket_ids=[]
    for bucket_id, bucket_id_data_set in enumerate(data_set):
        perm = np.array(xrange(len(bucket_id_data_set)))
        if reperm:#打乱一个bucket内部的数据顺序之后再分batch
            perm= np.random.permutation(len(bucket_id_data_set))
        repermed_bucket_id_data_set=[bucket_id_data_set[i] for i in perm]
        for i in xrange(len(repermed_bucket_id_data_set)/settings.batch_size):#看一个bucket能分成几个batch
            batch_data_sets.append([repermed_bucket_id_data_set[j] for j in range(i*settings.batch_size,(i+1)*settings.batch_size)])
            batch_bucket_ids.append(bucket_id)
    perm = np.array(xrange(len(batch_bucket_ids)))
    if reperm:#打乱输出的batch列表顺序
        perm = np.random.permutation(len(batch_bucket_ids))
    batch_data_sets = [batch_data_sets[i] for i in perm]
    batch_bucket_ids = [batch_bucket_ids[i] for i in perm]
    return (batch_data_sets,batch_bucket_ids)

if __name__ == '__main__':
    prepare_data(settings.data_dir)
    train_set = read_data('data/LCSTS/train.ids4000.sr','data/LCSTS/train.ids4000.tg')
    batch_datasets,batch_bucket_ids = batchize(train_set)
