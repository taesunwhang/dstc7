import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework import argsort

import os
import time
import logging
from datetime import datetime
import threading
import random


from data_process import DataProcess, AsyncDataSetWrapper
from data_helpers import load_vocab, load_trimmed_glove_vectors
from evaluate import evaluate_recall, mean_reciprocal_rank
from utils import TensorBoardSummaryWriter
from train_helper import average_gradients
from models import dual_encoder, esim_attention


class ESIMWord(object):
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


    def _inference(self, context, context_len, utterances, utterances_len):

        word_embeddings_init = tf.constant(self.word_embeddings, dtype=tf.float32)

        word_embeddings = tf.get_variable(
            name="word_embeddings",
            initializer=word_embeddings_init,
        )
        max_utterances_len = tf.shape(utterances)[2]

        context_embedded = tf.nn.embedding_lookup(word_embeddings, context)
        utterances_embedded = tf.nn.embedding_lookup(word_embeddings, utterances)

        """dual encoder"""
        reshaped_utterances_embedded = tf.reshape(utterances_embedded,
                                                  [-1, max_utterances_len, self.hparams.embedding_dim])
        dual_lstm_encoder = dual_encoder.DualEncoder(self.hparams, self.dropout_keep_prob_ph,
                                                     context_embedded, context_len,
                                                     reshaped_utterances_embedded, utterances_len)

        context_outputs = dual_lstm_encoder.context_outputs
        utterances_outputs = dual_lstm_encoder.utterances_outputs

        """esim_word_attention"""
        esim_word = esim_attention.ESIMAttention(self.hparams, self.dropout_keep_prob_ph,
                                                 context_outputs, context_len, utterances_outputs, utterances_len)
        # [batch_size, num_candidates, 64*6]
        esim_word_joined_feature = esim_word.context_word_joined_feature

        layer_out = tf.layers.dense(
            inputs=esim_word_joined_feature,
            units=self.hparams.rnn_hidden_dim,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.variance_scaling(
                scale=2.0, mode="fan_in", distribution="normal"),
            name="feed_forward_network"
        )

        logits = tf.layers.dense(
            inputs=layer_out,
            units=1,
            bias_initializer=tf.initializers.zeros(),
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="prediction"
        )

        logits = tf.squeeze(logits, [-1])

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
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss_op, global_step=self.global_step)

        eval = tf.nn.in_top_k(self.logits, self.target_ph, 1)
        correct_count = tf.reduce_sum(tf.cast(eval, tf.int32))

        self.accuracy = tf.divide(correct_count, tf.shape(self.target_ph)[0])
        self.predictions = argsort(self.logits, axis=1, direction='DESCENDING')

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
                data_process = DataProcess(self.hparams, self.input_path, "train", self.word2id)

                start_time = time.time()

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
                        accuracy_val, loss_val, global_step_val, _ = sess.run(
                            [self.accuracy,
                             self.loss_op,
                             self.global_step,
                             self.train_op],
                            feed_dict=self.make_feed_dict(batch_data_list[i], self.hparams.dropout_keep_prob)
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