import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework import argsort

import os
import time
import logging
from datetime import datetime
import random
import threading

from data_process import DataProcess, AsyncDataSetWrapper
from data_helpers import load_vocab, load_trimmed_glove_vectors
from evaluate import evaluate_recall, mean_reciprocal_rank
from utils import TensorBoardSummaryWriter, average_gradients

class DualEncoderLSTM(object):
    def __init__(self,hparams):
        self.hparams = hparams
        self.input_path = hparams.train_path
        self.get_embeddings(self.hparams.trimmed_glove_path)
        self.get_word2id(self.hparams.word_vocab_path)

        self._logger = logging.getLogger(__name__)

    def get_embeddings(self,filename):
        self.word_embeddings = load_trimmed_glove_vectors(filename)
        print(np.shape(self.word_embeddings))

    def get_word2id(self,filename):
        vocab, self.word2id = load_vocab(filename)
        print("%d train vocabulary word2id" % len(vocab))

    def _inference(self,context:tf.Tensor,context_len:tf.Tensor,utterances:tf.Tensor,utterances_len:tf.Tensor):

        #word_embeddings
        word_embeddings = tf.Variable(
            self.word_embeddings,
            name="word_embeddings",
            dtype=tf.float32,
            trainable=True
        )

        context_embedded = tf.nn.embedding_lookup(word_embeddings,context)

        with tf.variable_scope("context-encoder"):
            cell_context = tf.nn.rnn_cell.LSTMCell(self.hparams.rnn_hidden_dim,
                                                   forget_bias=2.0,
                                                   use_peepholes=True,
                                                   state_is_tuple=True)

            context_encoded_outputs, context_encoded_states = tf.nn.dynamic_rnn(
                cell_context,
                inputs=context_embedded,
                sequence_length=context_len,
                dtype=tf.float32)

        # [16,100,281,300]
        utterances_embedded = tf.nn.embedding_lookup(word_embeddings, utterances)
        utterances_embedded = tf.reshape(utterances_embedded,[-1,tf.shape(utterances)[-1],self.hparams.embedding_dim])
        utterances_len = tf.reshape(utterances_len,[-1])

        with tf.variable_scope("utterances-encoder"):
            cell_utterances = tf.nn.rnn_cell.LSTMCell(self.hparams.rnn_hidden_dim,
                                                      forget_bias=2.0,
                                                      use_peepholes=True,
                                                      state_is_tuple=True)
            #[16 100 MAX_LENGTH 300]
            utterances_encoded_outputs, utterances_encoded_states = tf.nn.dynamic_rnn(
                cell_utterances,
                inputs=utterances_embedded,
                sequence_length=utterances_len,
                dtype=tf.float32)

            all_utterances_encoded = tf.reshape(utterances_encoded_states[1],
                                                [tf.shape(utterances)[0],tf.shape(utterances)[1],
                                                 self.hparams.rnn_hidden_dim])

            # all_utteracnes_encoded(16, 100, 256)

        with tf.variable_scope("prediction"):
            M = tf.get_variable("M",
                                shape=[self.hparams.rnn_hidden_dim,self.hparams.rnn_hidden_dim],
                                initializer=tf.random_normal_initializer())

            #"predict" a response: c * M
            # context_encoded_states[1] -> [16 256]   /  M -> [256 256]
            generated_response = tf.matmul(context_encoded_states[1], M)
            # generated_response : [16 1 256]
            generated_response = tf.expand_dims(generated_response, axis=1)
            # all_utterances_encoded : [16 100 256] -> [16 256 100]
            all_utterances_encoded = tf.transpose(all_utterances_encoded, perm=[0, 2, 1])

            # logits : [16 1 256] * [16 256 100] -> [16 1 100]
            logits = tf.matmul(generated_response, all_utterances_encoded)
            logits = tf.squeeze(logits, [1])  # logits[16 100]

        return logits

    def make_placeholders(self):

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.context_ph = tf.placeholder(tf.int32, shape=[None, None], name="context")
        self.context_len_ph = tf.placeholder(tf.int32, shape=[None], name="context_len")

        self.utterances_ph = tf.placeholder(tf.int32, shape=[None, None, None], name="utterances")
        self.utterances_len_ph = tf.placeholder(tf.int32, shape=[None, None], name="utterances_len")

        self.target_ph = tf.placeholder(tf.int32, shape=[None], name="target")
        self.dropout_keep_prob_ph = tf.placeholder(tf.float32, shape=[], name="dropout_keep_prob")

        # [batch, num_sentences]
        self.context_sentence_ph = tf.placeholder(tf.int32, shape=[None, None, None], name="context_sentence")
        self.context_sentence_len_ph = tf.placeholder(tf.int32, shape=[None, None], name="context_sentence_len")
        self.tot_context_len_ph = tf.placeholder(tf.int32, shape=[None], name="tot_context_len")

        self.speaker_ph = tf.placeholder(tf.int32, shape=[None, None], name="speaker_ph")

    def build_train_graph(self):

        # logits
        with tf.variable_scope("inference", reuse=False):
            self.logits = self._inference(self.context_ph, self.context_len_ph,
                                          self.utterances_ph, self.utterances_len_ph)

        self.loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                      labels=self.target_ph,
                                                                      name="cross_entropy")

        self.loss_op = tf.reduce_mean(self.loss_op, name="cross_entropy_mean")
        # self.loss_op = self.loss_op + tf.reduce_mean(relevance_score)

        self.train_op = tf.train.AdamOptimizer().minimize(self.loss_op, global_step=self.global_step)

        eval = tf.nn.in_top_k(self.logits, self.target_ph, 1)
        correct_count = tf.reduce_sum(tf.cast(eval, tf.int32))

        self.accuracy = tf.divide(correct_count, tf.shape(self.target_ph)[0])
        self.predictions = argsort(self.logits, axis=1, direction='DESCENDING')

    def build_train_graph_multi_gpu(self):

        tower_grads = []

        context_ph = tf.split(self.context_ph, num_or_size_splits=len(self.hparams.gpu_num), axis=0)
        context_len_ph = tf.split(self.context_len_ph, num_or_size_splits=len(self.hparams.gpu_num), axis=0)
        utterances_ph = tf.split(self.utterances_ph, num_or_size_splits=len(self.hparams.gpu_num), axis=0)
        utterances_len_ph = tf.split(self.utterances_len_ph, num_or_size_splits=len(self.hparams.gpu_num), axis=0)
        target_ph = tf.split(self.target_ph, num_or_size_splits=len(self.hparams.gpu_num), axis=0)

        optimizer = tf.train.AdamOptimizer(self.hparams.learning_rate)
        for i, gpu_id in enumerate(self.hparams.gpu_num):
            with tf.device('/gpu:%d' % gpu_id):
                with tf.variable_scope("inference", reuse=tf.AUTO_REUSE):
                    logits = self._inference(context_ph[i], context_len_ph[i],
                                             utterances_ph[i], utterances_len_ph[i])

                loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                         labels=target_ph[i],
                                                                         name="cross_entropy")
                loss_op = tf.reduce_mean(loss_op, name="cross_entropy_mean")

                if i == 0:
                    self.loss_op = loss_op
                    self.logits = logits

                    eval = tf.nn.in_top_k(logits, target_ph[i], 1)
                    correct_count = tf.reduce_sum(tf.cast(eval, tf.int32))
                    self.accuracy = tf.divide(correct_count, tf.shape(target_ph[i])[0])

                    self.predictions = argsort(self.logits, axis=1, direction='DESCENDING')

                grads = optimizer.compute_gradients(loss_op)
                # tf.get_variable_scope().reuse_variables()
                tower_grads.append(grads)

        grads = average_gradients(tower_grads)
        self.train_op = optimizer.apply_gradients(grads, self.global_step)

    def make_feed_dict(self, pad_batch_data, dropout_keep_prob):

        feed_dict={}
        (pad_context, context_len_batch), \
        (pad_utterances, utterances_len_batch), target_batch, \
        (pad_context_sentence, context_sentence_len, tot_context_len), speaker_batch, \
        example_id_batch, candidates_id_batch = pad_batch_data


        feed_dict[self.context_ph] = pad_context
        feed_dict[self.context_len_ph] = context_len_batch
        feed_dict[self.utterances_ph] = pad_utterances
        feed_dict[self.utterances_len_ph] = utterances_len_batch
        feed_dict[self.target_ph] = target_batch
        feed_dict[self.dropout_keep_prob_ph] = dropout_keep_prob

        feed_dict[self.context_sentence_ph] = pad_context_sentence
        feed_dict[self.context_sentence_len_ph] = context_sentence_len
        feed_dict[self.tot_context_len_ph] = tot_context_len

        feed_dict[self.speaker_ph] = speaker_batch

        return feed_dict

    def ubuntu_train(self, saved_file=None):

        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
        )
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        with sess.as_default():
            self.make_placeholders()

            # multiple_gpu
            if len(self.hparams.gpu_num) > 1:
                self.build_train_graph_multi_gpu()

            else:
                # single gpu
                with tf.device("/gpu:%d" % self.hparams.gpu_num[0]):
                    self.build_train_graph()

            # Tensorboard
            tensorboard_summary = TensorBoardSummaryWriter(self.hparams.root_dir, sess, sess.graph)

            tf.global_variables_initializer().run()
            saver = tf.train.Saver(max_to_keep=30)

            if saved_file is not None:
                saver.restore(sess, saved_file)

            start_time = datetime.now().strftime('%H:%M:%S')
            print("Start train model at %s" % start_time)

            step_loss_mean, step_accuracy_mean = 0, 0

            for epoch_completed in range(self.hparams.num_epochs):
                # tf.reset_default_graph()
                data_process = DataProcess(self.hparams, self.input_path, "train", self.word2id)

                start_time = time.time()

                # batch_size, num_candidates = self.curriculum_learning(epoch_completed)
                while True:

                    pad_batch_data = data_process.get_batch_data(self.hparams.batch_size,
                                                                 self.hparams.num_candidates)

                    if pad_batch_data is None:
                        break

                    (context, _), (utterances, _), _, _, _, _, _ = pad_batch_data

                    accuracy_val, loss_val, global_step_val, _ = sess.run(
                        [self.accuracy,
                         self.loss_op,
                         self.global_step,
                         self.train_op],
                        feed_dict=self.make_feed_dict(pad_batch_data, self.hparams.dropout_keep_prob)
                    )

                    step_loss_mean += loss_val
                    step_accuracy_mean += accuracy_val

                    if global_step_val % 100 == 0:
                        step_loss_mean /= 100
                        step_accuracy_mean /= 100

                        tensorboard_summary.add_summary("train/cross_entropy", step_loss_mean, global_step_val)
                        tensorboard_summary.add_summary("train/accuracy", step_accuracy_mean, global_step_val)

                        self._logger.info("[Step %d][%d th] loss: %.4f, accuracy: %.2f%%  (%.2f seconds)" % (
                            global_step_val,
                            data_process.index,
                            step_loss_mean,
                            step_accuracy_mean * 100,
                            time.time() - start_time))

                        step_loss_mean, step_accuracy_mean = 0, 0
                        start_time = time.time()

                        # Tensorboard
                        if global_step_val % self.hparams.dev_evaluate_step == 0:
                            k_list, recall_res, avg_mrr = self.run_evaluate(sess, "dev", self.hparams.valid_path)

                            for i, k in enumerate(k_list):
                                tensorboard_summary.add_summary(
                                    "dev_recall/recall_%s" % k, float(recall_res[i]), global_step_val)
                                tensorboard_summary.add_summary(
                                    "dev_mrr/mean_reciprocal_error", float(avg_mrr), global_step_val)

                            k_list, recall_res, avg_mrr = self.run_evaluate(sess, "test", self.hparams.test_path)
                            for i, k in enumerate(k_list):
                                tensorboard_summary.add_summary(
                                    "test_recall/recall_%s" % k, float(recall_res[i]), global_step_val)
                                tensorboard_summary.add_summary(
                                    "test_mrr/mean_reciprocal_error", float(avg_mrr), global_step_val)

                            save_path = saver.save(sess,
                                                   os.path.join(self.hparams.root_dir,
                                                                "%.2f-model.ckpt" % recall_res[0]),
                                                   global_step=global_step_val)

                self._logger.info("End of epoch %d." % (epoch_completed + 1))

            tensorboard_summary.close()

            if sess is not None:
                sess.close()

    def advising_train(self, saved_file=None):
        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
        )
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        with sess.as_default():
            self.make_placeholders()

            # multiple_gpu
            if len(self.hparams.gpu_num) > 1:
                self.build_train_graph_multi_gpu()

            else:
                # single gpu
                with tf.device("/gpu:%d" % self.hparams.gpu_num[0]):
                    self.build_train_graph()

            # Tensorboard
            tensorboard_summary = TensorBoardSummaryWriter(self.hparams.root_dir, sess, sess.graph)

            tf.global_variables_initializer().run()
            saver = tf.train.Saver(max_to_keep=30)

            if saved_file is not None:
                saver.restore(sess, saved_file)


            step_loss_mean, step_accuracy_mean = 0, 0


            lock = threading.Lock()
            event = threading.Event()

            batch_data_thread = AsyncDataSetWrapper(self.hparams, self.input_path, self.hparams.data_pickle,
                                                    self.word2id,self.hparams.batch_size, self.hparams.num_candidates,
                                                    lock, event, word_embedding_type="glove")

            start_time = datetime.now().strftime('%H:%M:%S')
            print("Start train model at %s" % start_time)
            batch_data_list = []
            for epoch_completed in range(self.hparams.num_epochs):

                start_time = time.time()

                batch_data_list = batch_data_thread.get_batch_data().copy()

                event.set()
                if len(batch_data_list) < 10000:
                    event.set()
                    print("There is no train batch data yet.. Need to load Traind data set")
                    while True:
                        # print("Getting batch data:", len(batch_data_thread.get_batch_data()))
                        time.sleep(10)
                        if len(batch_data_thread.get_batch_data()) == 10000:
                            break

                    # waiting for Thread to get batch_data
                    batch_data_list = batch_data_thread.get_batch_data()

                print("batch_data length:", len(batch_data_list))
                if len(batch_data_list) % 10000 == 0:

                    #shuffle
                    print("data shuffling has started")
                    random.shuffle(batch_data_list)

                    for i in range(len(batch_data_list)):
                        # gpu_start_time = time.time()
                        accuracy_val, loss_val, global_step_val, _ = sess.run(
                            [self.accuracy,
                             self.loss_op,
                             self.global_step,
                             self.train_op],
                            feed_dict=self.make_feed_dict(batch_data_list[i], self.hparams.dropout_keep_prob)
                        )
                        # gpu_finsih_time = time.time()
                        # print("gpu_time %.2f" % (gpu_finsih_time-gpu_start_time))

                        step_loss_mean += loss_val
                        step_accuracy_mean += accuracy_val

                        if global_step_val % 100 == 0:
                            step_loss_mean /= 100
                            step_accuracy_mean /= 100

                            tensorboard_summary.add_summary("train/cross_entropy", step_loss_mean, global_step_val)
                            tensorboard_summary.add_summary("train/accuracy", step_accuracy_mean, global_step_val)

                            self._logger.info("[Step %d][%d th] loss: %.4f, accuracy: %.2f%%  (%.2f seconds)" % (
                                global_step_val,
                                i+1,
                                step_loss_mean,
                                step_accuracy_mean * 100,
                                time.time() - start_time))

                            step_loss_mean, step_accuracy_mean = 0, 0
                            start_time = time.time()

                        # Tensorboard
                        if global_step_val % self.hparams.dev_evaluate_step == 0:
                            k_list, recall_res, avg_mrr = self.run_evaluate(sess, "dev", self.hparams.valid_path)

                            for i, k in enumerate(k_list):
                                tensorboard_summary.add_summary(
                                    "recall_dev/recall_%s" % k, float(recall_res[i]), global_step_val)
                                tensorboard_summary.add_summary(
                                    "mrr_dev/mean_reciprocal_error", float(avg_mrr), global_step_val)

                            k_list, recall_res, avg_mrr = self.run_evaluate(sess, "test", self.hparams.test_path, 1)
                            for i, k in enumerate(k_list):
                                tensorboard_summary.add_summary(
                                    "recall_test1/recall_%s" % k, float(recall_res[i]), global_step_val)
                                tensorboard_summary.add_summary(
                                    "mrr_test1/mean_reciprocal_error", float(avg_mrr), global_step_val)

                            k_list, recall_res, avg_mrr = self.run_evaluate(sess, "test", self.hparams.test2_path, 2)
                            for i, k in enumerate(k_list):
                                tensorboard_summary.add_summary(
                                    "recall2/recall_%s" % k, float(recall_res[i]), global_step_val)
                                tensorboard_summary.add_summary(
                                    "mrr2/mean_reciprocal_error", float(avg_mrr), global_step_val)

                            save_path = saver.save(sess,
                                                   os.path.join(self.hparams.root_dir,
                                                                "test2-%.2f-model.ckpt" % recall_res[0]),
                                                   global_step=global_step_val)

                            self._logger.info("Model saved at : %s" % save_path)

                self._logger.info("End of epoch %d." % (epoch_completed + 1))

            tensorboard_summary.close()
            batch_data_thread.kill()

            if sess is not None:
                sess.close()

    def run_evaluate(self, sess, type, data_path, test_case=1):
        data_process = DataProcess(self.hparams, data_path, type, word2id=self.word2id, test_case=test_case)

        k_list = self.hparams.recall_k_list
        total_examples = 0
        total_correct = np.zeros([len(k_list)], dtype=np.int32)
        total_mrr = 0

        index = 0

        while True:
            batch_data = data_process.get_batch_data(self.hparams.dev_batch_size, 100)

            if batch_data is None:
                break

            (context, _), (utterances, _), _, _, _, example_id, candidates_id = batch_data

            pred_val, _ = sess.run([self.predictions, self.logits],
                                   feed_dict=self.make_feed_dict(batch_data, 1.0))

            pred_val = np.asarray(pred_val)
            num_correct, num_examples = evaluate_recall(pred_val, batch_data[2], k_list)
            total_mrr += mean_reciprocal_rank(pred_val, batch_data[2])

            total_examples += num_examples
            total_correct = np.add(total_correct, num_correct)

            # if num_correct[5] != self.hparams.dev_batch_size:
            #     print(example_id, ":", index, num_correct[5])

            index += 1
            if index % 500 == 0:
                accumulated_accuracy = (total_correct / total_examples) * 100
                print("index : ", index, " | ", accumulated_accuracy)

        avg_mrr = total_mrr / (self.hparams.dev_batch_size * index)
        recall_result = ""

        for i in range(len(k_list)):
            recall_result += "Recall@%s : " % k_list[i] + "%.2f%% | " % ((total_correct[i] / total_examples) * 100)
        self._logger.info(recall_result)
        self._logger.info("MRR: %.4f" % avg_mrr)

        return k_list, (total_correct / total_examples) * 100, avg_mrr

    def evaluate(self, saved_file:str):

        context = tf.placeholder(tf.int32, shape=[None, None], name="context")
        context_len = tf.placeholder(tf.int32, shape=[None], name="context_len")
        utterances = tf.placeholder(tf.int32, shape=[None, None, None], name="utterances")
        utterances_len = tf.placeholder(tf.int32, shape=[None, None], name="utterances_len")
        target = tf.placeholder(tf.int32, shape=[None], name="target")

        # logits
        with tf.variable_scope("inference", reuse=False):
            logits = self._inference(context, context_len, utterances, utterances_len)

        predictions = argsort(logits, axis=1,direction='DESCENDING')

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(sess, saved_file)

        data = DataProcess(self.hparams.valid_path, "test", self.word2id)

        k_list = [1,2,5,10,50,100]
        total_examples = 0
        total_correct = np.zeros([6],dtype=np.int32)

        while True:
            pad_batch_data = data.get_batch_data(self.hparams.batch_size)

            if pad_batch_data is None:
                break
            (pad_context, context_len_batch), (pad_utterances, utterances_len_batch), target_batch = pad_batch_data

            feed_dict = {context: pad_context, context_len: context_len_batch,
                         utterances: pad_utterances, utterances_len: utterances_len_batch,
                         target: target_batch}
            pred_val = sess.run([predictions], feed_dict=feed_dict)

            pred_val = np.asarray(pred_val).squeeze(0)
            num_correct, num_examples = evaluate_recall(pred_val,target_batch, k_list)

            total_examples += num_examples
            total_correct = np.add(total_correct, num_correct)

        recall_result = ""
        for i in range(len(k_list)):
            recall_result += "Recall@%s : " % k_list[i] + "%.2f%% | " % ((total_correct[i] / total_examples) * 100)
        self._logger.info(recall_result)
