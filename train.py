#!/usr/bin/env python
#-*-coding:utf-8-*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import data_utils
import seq2seqmodel
import settings
import tensorflow as tf
import numpy as np
import time
import datetime
import utils
import os

current_step = 0
current_epoch = 0


def initialize_variable(session, model, variable_name, value):
    """根据variable_name给模型中的变量赋值"""
    #embedding_variable_name = "RNN/EmbeddingWrapper/embedding"
    vs = [v for v in tf.all_variables() if variable_name in v.name]
    if len(vs) == 0:
        print 'no variable', variable_name, ' in the model to initialize'
        sys.exit(1)
    elif len(vs) > 1:
        print 'too much variables contain', variable_name
        sys.exit(1)
    else:
        tf_variable = vs[0]
        session.run([tf_variable.initializer], {
                    tf_variable.initializer.inputs[1]: value})


def pre_train_word2vec():
    """预训练word2vec"""
    encode_texts = []
    decode_texts = []
    for bucket_id, pairs in enumerate(data_utils.data_set):
        shoud_break = False
        for source, target in pairs:
            encode_texts.append(data_utils.token_ids_to_wordlist(source))
            decode_texts.append(data_utils.token_ids_to_wordlist(target))
            if len(encode_texts) == settings.initialize_embedding_data_num:
                shoud_break = True
                break
        if shoud_break:
            break
    encode_w2v = np.random.uniform(-0.1, 0.1,
                                   size=(len(data_utils.inv_dictionary), settings.embedding_size))
    decode_w2v = np.random.uniform(-0.1, 0.1,
                                   size=(len(data_utils.inv_dictionary), settings.embedding_size))
    encode_model = gensim.models.Word2Vec(encode_texts, size, workers=10)
    decode_model = gensim.models.Word2Vec(decode_texts, size, workers=10)
    for word in encode_model.vocab:
        for k in xrange(size):
            encode_w2v[data_utils.dictionary[word], k] = encode_model[word][k]
    for word in decode_model.vocab:
        for k in xrange(size):
            decode_w2v[data_utils.dictionary[word], k] = decode_model[word][k]
    return (encode_w2v, decode_w2v)


def create_model(session, forward_only):
    """创建模型，并初始化参数"""
    model = seq2seqmodel.Model(
        vocab_size=len(data_utils.dictionary),
        buckets=settings.buckets,
        embedding_size=settings.embedding_size,
        hidden_size=settings.hidden_size,
        num_layers=settings.num_layers,
        batch_size=settings.batch_size,
        cell_type=settings.cell_type,
        num_samples=settings.num_samples,
        forward_only=forward_only
    )
    ckpt = tf.train.get_checkpoint_state(settings.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Initilizing model with fresh parameters.")
        session.run(tf.initialize_all_variables())
        if settings.initialize_embedding:
            print("Initilizing embedding with word2vec.")
            encode_w2v, decode_w2v = pre_train_word2vec()
            initialize_variable(
                session, model, 'EmbeddingWrapper/embedding', encode_w2v)
            initialize_variable(
                session, model, '_decoder/embedding:0', decode_w2v)
    #print [v.name for v in tf.all_variables() if 'embedding' in v.name]
    #print tf.get_default_graph().get_tensor_by_name(u'embedding_rnn_seq2seq/RNN/EmbeddingWrapper/embedding:0').eval().shape
    #print tf.get_default_graph().get_tensor_by_name(u'embedding_rnn_seq2seq/embedding_rnn_decoder/embedding:0').eval().shape
    return model


def do_batch(model, session, batch_data, bucket_id):
    global current_step, previous_batch_losses
    current_step += 1
    # 执行batch
    d1 = datetime.datetime.now()
    encoder_inputs, decoder_inputs, target_weights = model.transpose_batch(
        batch_data, bucket_id)
    _, step_loss, _ = model.step(
        session, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
    d2 = datetime.datetime.now()
    if current_step%settings.step_per_testpoint==0:
        do_test(session,model)
    return (step_loss, d2 - d1)


def do_epoch(model, session):
    """每个epoch重新划分一次batch"""
    global current_epoch
    current_epoch += 1
    # 划分batch
    batch_datas = []
    batch_bucket_ids = []
    for bucket_id, bucket_data_set in enumerate(data_utils.data_set):
        for one in utils.shuffle_divide(bucket_data_set, settings.batch_size):
            batch_datas.append(one)
            batch_bucket_ids.append(bucket_id)
    batch_num = len(batch_datas)
    # 乱序执行batch
    perm = np.random.permutation(batch_num)
    batch_num = 0
    batch_times = []
    batch_losses = []
    epoch_loss = 0.0
    for batch_id in perm:
        batch_num+=1
        print 'epoch:'+str(current_epoch)+'/'+str(settings.max_epoch)+'\t'+'batch:'+str(batch_num)+'/'+str(len(perm)),
        batch_loss, batch_time = do_batch(model, session, batch_datas[
                                          batch_id], batch_bucket_ids[batch_id])
        batch_times.append(batch_time)
        batch_losses.append(batch_loss)
        time_used = np.sum(batch_times)
        time_eta = time_used / len(batch_times) * \
            (len(batch_datas) - len(batch_times))
        epoch_loss += batch_loss
        mean_batch_loss = epoch_loss / len(batch_losses)
        print '\tbatch loss:'+str(batch_loss),
        print '\tbatchtime:'+str(round(batch_time.total_seconds(),1))+' epochtime:'+str(round(time_used.total_seconds(),1))+'/'+str(round((time_used+time_eta).total_seconds(),1            ))
        sys.stdout.flush()
    return epoch_loss


def do_test(session, model):
    for i, wordlist in enumerate(data_utils.test_data[0:10]):
        test_source = data_utils.wordlist_to_token_ids(wordlist)
        test_target = model.decode(session, test_source)
        test_target_words = ' '.join(
            [data_utils.inv_dictionary[wid] for wid in test_target]).encode('utf-8')
        print test_target_words
        sys.stdout.flush()
    print ''
    sys.stdout.flush()


def do_checkpoint(session, model):
    checkpoint_path = os.path.join(settings.train_dir, "train.ckpt")
    model.saver.save(session, checkpoint_path, global_step=current_step)


def train():
    data_utils.load_and_prepare_data()
    with tf.Session() as sess:
        print("Creating %d layers of %d units." %
              (settings.num_layers, settings.hidden_size))
        model = create_model(sess, settings.forward_only)
        while True:
            loss = 0
            loss = do_epoch(model, sess)
            do_checkpoint(sess, model)
            if current_epoch == settings.max_epoch:
                break

if __name__ == '__main__':
    train()
