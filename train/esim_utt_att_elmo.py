import os
import tensorflow as tf
import numpy as np
# np.set_printoptions(threshold=np.nan)
from tensorflow.contrib import rnn
from tensorflow.contrib.framework import argsort

import os
import logging
from datetime import datetime
import time
import json
import random
import threading
import math

from models import elmo_embedding, dual_encoder, esim_attention, sentence_bilinear_attention
from data_process import DataProcess, AsyncDataSetWrapper
from data_helpers import load_vocab, load_trimmed_glove_vectors
from evaluate import evaluate_recall, mean_reciprocal_rank
from utils import TensorBoardSummaryWriter, average_gradients

class ESIMUttAttElmo(object):
    def __init__(self, hparams):

        self.hparams = hparams
        self.input_path = hparams.train_path

        self.get_embeddings(self.hparams.trimmed_glove_path)
        self.get_vocab(self.hparams.word_vocab_path)

        self._logger = logging.getLogger(__name__)


    def get_embeddings(self,filename):
        self.word_embeddings = load_trimmed_glove_vectors(filename)
        self.word_embeddings_shape = np.shape(self.word_embeddings)

        print(np.shape(self.word_embeddings))

    def get_vocab(self, word_vocab_path):
        word_vocab, self.word2id = load_vocab(word_vocab_path)
        print("%d word vocabulary word2id" % len(word_vocab))

    def _inference(self, context_embedded, context_len, utterances_embedded, utterances_len,
                   context_sentence_embedded, context_sentence_len, tot_context_len, speaker):

        batch_size = tf.shape(context_embedded)[0]
        max_utterances_len = tf.shape(utterances_embedded)[1]
        max_context_sentence_len = tf.shape(context_sentence_embedded)[1]

        """dual encoder"""
        dual_lstm_encoder = dual_encoder.DualEncoder(self.hparams, self.dropout_keep_prob_ph,
                                                     context_embedded, context_len, utterances_embedded, utterances_len)

        context_outputs = dual_lstm_encoder.context_outputs
        utterances_outputs = dual_lstm_encoder.utterances_outputs

        """esim word"""
        esim_word = esim_attention.ESIMAttention(self.hparams, self.dropout_keep_prob_ph,
                                                 context_outputs, context_len, utterances_outputs, utterances_len)

        word_joined_feature = esim_word.context_word_joined_feature

        """Context Sentence Attention"""
        # Context Sentence Attention
        utterances_embedded = tf.reshape(utterances_embedded,
                                         [batch_size, -1, max_utterances_len, self.hparams.embedding_dim])
        context_sentence_embedded = tf.reshape(context_sentence_embedded,
                                               [batch_size, -1, max_context_sentence_len, self.hparams.embedding_dim])

        sentence_attention = sentence_bilinear_attention.SentenceBilinearAtt(
            self.hparams, self.dropout_keep_prob_ph,
            context_sentence_embedded, context_sentence_len, tot_context_len,
            utterances_embedded, utterances_len, speaker)
        sentence_attention.pos_emb_bool = True
        sentence_attention.user_emb_bool = True
        sentence_joined_feature = sentence_attention.get_sentence_joined_feature()

        joined_feature = tf.concat(axis=-1, values=[word_joined_feature, sentence_joined_feature])

        feed_forward_outputs = tf.layers.dense(
            inputs=joined_feature,
            units=self.hparams.rnn_hidden_dim,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.variance_scaling(
                scale=1.0, mode="fan_avg", distribution="normal"),
            name="FFNN"
        )

        logits = tf.layers.dense(
            inputs=feed_forward_outputs,
            units=1,
            bias_initializer=tf.initializers.zeros(),
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="prediction"
        )

        logits = tf.squeeze(logits, [-1])

        return logits

    def make_placeholders(self):
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        # elmo_embedding
        self.context_ids_ph, self.utterances_ids_ph, self.context_sentence_ids_ph = \
            self.elmo.make_placeholders()

        #[batch_size, max_context_len]
        self.context_len_ph = tf.placeholder(tf.int32, shape=[None], name="context_len")

        #[batch_size*num_candidates, max_response_len, 256]
        self.utterances_len_ph = tf.placeholder(tf.int32, shape=[None, None], name="utterances_len")

        self.target_ph = tf.placeholder(tf.int32, shape=[None], name="target")
        self.dropout_keep_prob_ph = tf.placeholder(tf.float32, shape=[], name="dropout_keep_prob")

        #[batch, max_context_len, max_sentence_len, 256]
        self.context_sentence_len_ph = tf.placeholder(tf.int32, shape=[None, None], name="context_sentence_len")
        self.tot_context_len_ph = tf.placeholder(tf.int32, shape=[None], name="tot_context_len")

        self.speaker_ph = tf.placeholder(tf.int32, shape=[None, None], name="speaker_ph")

    def build_train_graph(self):
        (elmo_context_input, elmo_utterances_input, elmo_context_sentence_input) = \
            self.elmo.build_embeddings_op(self.context_ids_ph, self.utterances_ids_ph,
                                          self.context_sentence_ids_ph)

        # logits
        with tf.variable_scope("inference", reuse=False):
            self.logits = self._inference(elmo_context_input['weighted_op'], self.context_len_ph,
                                          elmo_utterances_input['weighted_op'], self.utterances_len_ph,
                                          elmo_context_sentence_input['weighted_op'], self.context_sentence_len_ph,
                                          self.tot_context_len_ph, self.speaker_ph)

        self.loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                      labels=self.target_ph,
                                                                      name="cross_entropy")
        # self.logits_max = tf.argmax(self.logits, axis=-1)
        self.loss_op = tf.reduce_mean(self.loss_op, name="cross_entropy_mean")
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss_op, global_step=self.global_step)

        eval = tf.nn.in_top_k(self.logits, self.target_ph, 1)
        correct_count = tf.reduce_sum(tf.cast(eval, tf.int32))

        self.accuracy = tf.divide(correct_count, tf.shape(self.target_ph)[0])
        self.predictions = argsort(self.logits, axis=1, direction='DESCENDING')
        self.confidence = tf.nn.softmax(self.logits, axis=-1)

    def build_train_graph_multi_gpu(self):

        gpu_num = len(self.hparams.gpu_num)

        context_ids_ph = tf.split(self.context_ids_ph, gpu_num, 0)
        context_len_ph = tf.split(self.context_len_ph, gpu_num, 0)

        utterances_ids_ph = tf.split(self.utterances_ids_ph, gpu_num, 0)
        utterances_len_ph = tf.split(self.utterances_len_ph, gpu_num, 0)

        target_ph = tf.split(self.target_ph, gpu_num, 0)

        context_sentence_ids_ph = tf.split(self.context_sentence_ids_ph, gpu_num, 0)
        context_sentence_len_ph = tf.split(self.context_sentence_len_ph, gpu_num, 0)
        tot_context_len_ph = tf.split(self.tot_context_len_ph, gpu_num, 0)
        speaker_ph = tf.split(self.speaker_ph, gpu_num, 0)

        optimizer = tf.train.AdamOptimizer(self.hparams.learning_rate)

        tower_grads = []
        tot_losses = []
        tot_logits = []
        tot_labels = []

        for i, gpu_id in enumerate(self.hparams.gpu_num):
            with tf.device('/gpu:%d' % gpu_id):
                with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                    (elmo_context_input, elmo_utterances_input, elmo_context_sentence_input) = \
                        self.elmo.build_embeddings_op(context_ids_ph[i], utterances_ids_ph[i],
                                                      context_sentence_ids_ph[i])

                with tf.variable_scope("inference", reuse=tf.AUTO_REUSE):
                    logits = self._inference(elmo_context_input['weighted_op'], context_len_ph[i],
                                             elmo_utterances_input['weighted_op'], utterances_len_ph[i],
                                             elmo_context_sentence_input['weighted_op'], context_sentence_len_ph[i],
                                             tot_context_len_ph[i], speaker_ph[i])

                loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                         labels=target_ph[i],
                                                                         name="cross_entropy")
                loss_op = tf.reduce_mean(loss_op, name="cross_entropy_mean")

                tot_losses.append(loss_op)
                tot_logits.append(logits)
                tot_labels.append(target_ph[i])

                grads = optimizer.compute_gradients(loss_op)
                tower_grads.append(grads)
                # tf.get_variable_scope().reuse_variables()

        grads = average_gradients(tower_grads)
        self.loss_op = tf.divide(tf.add_n(tot_losses), gpu_num)
        self.logits = tf.concat(tot_logits, axis=0)
        tot_labels = tf.concat(tot_labels, axis=0)
        self.train_op = optimizer.apply_gradients(grads, self.global_step)

        eval = tf.nn.in_top_k(self.logits, tot_labels, 1)
        correct_count = tf.reduce_sum(tf.cast(eval, tf.int32))
        self.accuracy = tf.divide(correct_count, tf.shape(self.target_ph)[0])
        self.predictions = argsort(self.logits, axis=1, direction='DESCENDING')
        self.confidence = tf.nn.softmax

    def make_feed_dict(self, batch_data, dropout_keep_prob):

        feed_dict={}
        (context_batch, context_len_batch), \
        (utterances_batch, utterances_len_batch), target_batch, \
        (context_sentence_batch, context_sentence_len_batch, tot_context_len), speaker_batch, \
        example_id, candidates_id = batch_data

        context_ids, utterances_ids, context_sentence_ids = \
            self.elmo.get_toknized_data(context_batch, utterances_batch, context_sentence_batch)

        feed_dict[self.context_ids_ph] = context_ids
        feed_dict[self.context_len_ph] = context_len_batch

        feed_dict[self.utterances_ids_ph] = utterances_ids
        feed_dict[self.utterances_len_ph] = utterances_len_batch
        feed_dict[self.target_ph] = target_batch
        feed_dict[self.dropout_keep_prob_ph] = dropout_keep_prob

        feed_dict[self.context_sentence_ids_ph] = context_sentence_ids
        feed_dict[self.context_sentence_len_ph] = context_sentence_len_batch
        feed_dict[self.tot_context_len_ph] = tot_context_len
        feed_dict[self.speaker_ph] = speaker_batch

        return feed_dict

    def ubuntu_train(self, saved_file = None):

        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.8
        sess = tf.Session(config=config)


        with sess.as_default():

            self.elmo = elmo_embedding.ELMoEmbeddings(self.hparams)
            self.make_placeholders()

            # multiple_gpu
            if len(self.hparams.gpu_num) > 1:
                self.build_train_graph_multi_gpu()

            else:
                # single gpu
                with tf.device("/gpu:%d" % self.hparams.gpu_num[0]):
                    self.build_train_graph()

            tf.global_variables_initializer().run()

            # Tensorboard
            tensorboard_summary = TensorBoardSummaryWriter(self.hparams.root_dir, sess, sess.graph)

            saver = tf.train.Saver(max_to_keep=30)

            if saved_file is not None:
                saver.restore(sess, saved_file)

            start_time = datetime.now().strftime('%H:%M:%S')
            print("Start train model at %s" % start_time)

            step_loss_mean, step_accuracy_mean = 0, 0

            for epoch_completed in range(self.hparams.num_epochs):

                data_process = DataProcess(self.hparams,  self.input_path, "train", self.word2id)
                start_time = time.time()

                while True:
                    batch_data = data_process.get_batch_data_elmo(self.hparams.batch_size, self.hparams.num_candidates)

                    if batch_data is None:
                        break

                    (context, _), (utterances, _), _, _, _, example_id, candidates_id = batch_data

                    accuracy_val, loss_val, global_step_val, _ = sess.run(
                        [self.accuracy,
                         self.loss_op,
                         self.global_step,
                         self.train_op],
                        feed_dict=self.make_feed_dict(batch_data, self.hparams.dropout_keep_prob)
                    )

                    step_loss_mean += loss_val
                    step_accuracy_mean += accuracy_val

                    if global_step_val % self.hparams.tensorboard_step == 0:

                        step_loss_mean /= self.hparams.tensorboard_step
                        step_accuracy_mean /= self.hparams.tensorboard_step

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

                    #Tensorboard
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
                                               os.path.join(self.hparams.root_dir, "%.2f-model.ckpt" % recall_res[0]),
                                               global_step=global_step_val)

                        self._logger.info("Model saved at : %s" % save_path)

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
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        sess = tf.Session(config=config)

        with sess.as_default():
            self.elmo = elmo_embedding.ELMoEmbeddings(self.hparams)
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
                                                    self.word2id, self.hparams.batch_size, self.hparams.num_candidates,
                                                    lock, event, word_embedding_type="elmo")

            total_data_len = math.ceil(100000 / self.hparams.batch_size)
            print("total_data_len is %d" % total_data_len)
            for epoch_completed in range(self.hparams.num_epochs):

                batch_data_list = batch_data_thread.get_batch_data().copy()

                print("Waiting.. until Thread is ready for get data!")
                time.sleep(10)
                print("Sent an event to waiting Thread!")
                event.set()
                if len(batch_data_list) < total_data_len:
                    print("There is no train batch data yet.. Need to load Traind data set")
                    while True:
                        # print("Getting batch data:", len(batch_data_thread.get_batch_data()))
                        time.sleep(10)
                        if len(batch_data_thread.get_batch_data()) == total_data_len:
                            break

                    # waiting for Thread to get batch_data
                    batch_data_list = batch_data_thread.get_batch_data().copy()

                print("batch_data length:", len(batch_data_list))
                if len(batch_data_list) % total_data_len == 0:

                    # shuffle
                    print("data shuffling has started")
                    random.shuffle(batch_data_list)
                    event.clear()

                    print("Start train model at %s" % datetime.now().strftime('%H:%M:%S'))
                    start_time = time.time()

                    for i in range(len(batch_data_list)):
                        gpu_start_time = time.time()
                        accuracy_val, loss_val, global_step_val, _ = sess.run(
                            [self.accuracy,
                             self.loss_op,
                             self.global_step,
                             self.train_op],
                            feed_dict=self.make_feed_dict(batch_data_list[i], self.hparams.dropout_keep_prob)
                        )
                        gpu_finsih_time = time.time()
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
                                i + 1,
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
            batch_data = data_process.get_batch_data_elmo(self.hparams.dev_batch_size, 100)

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


    def evaluate(self, saved_file: str):

        self.elmo = elmo_embedding.ELMoEmbeddings(self.hparams)
        self.make_placeholders()
        self.build_train_graph()

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(sess, saved_file)

        data_process = DataProcess(self.hparams.test2_path, "test", self.word2id)
        #open test results file
        print("test path : ", self.hparams.test2_path)


        k_list = [1, 2, 5, 10, 50, 100]
        with open(self.hparams.test_result, "w", encoding="utf-8") as f_handle:
            example_list = []
            test_batch_index = 0
            print("Start making test results json file")
            while True:
                batch_data = data_process.get_batch_data_elmo(self.hparams.dev_batch_size, 100)

                if batch_data is None:
                    break

                (context, _), (utterances, _), target_id, _, _, example_id, candidates_id = batch_data

                pred_val, confidence_val, _ = sess.run([self.predictions, self.confidence, self.logits],
                                                       feed_dict=self.make_feed_dict(batch_data, 1.0))
                pred_val = np.squeeze(pred_val, axis=0)
                confidence_val = np.squeeze(confidence_val, axis=0)
                candidates_id = candidates_id[0]
                example_id = example_id[0]
                # print(utterances[0][target_id[0]])
                test_batch_index += 1
                print(example_id, ":", test_batch_index, "[th] predicted answer : ", " ".join(utterances[0][pred_val[0]]))
                #100 candidates confidence level
                candidate_list = []
                for pred_index in pred_val:
                    candidate_list.append(
                        {"candidate-id": candidates_id[pred_index], "confidence": float(confidence_val[pred_index])}
                    )

                example_result = {"example-id" : example_id, "candidate-ranking" : candidate_list}
                example_list.append(example_result)

            json_result = json.dumps(example_list, indent=4)
            f_handle.write(json_result)

