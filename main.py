import argparse
import collections
import logging
from datetime import datetime
import json
import os

from train import dual_encoder_LSTM, esim_word, esim_sentence_att, esim_utt_att_glove, esim_utt_att_elmo

def init_logger(path:str):
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    debug_fh = logging.FileHandler(os.path.join(path, "debug.log"))
    debug_fh.setLevel(logging.DEBUG)

    info_fh = logging.FileHandler(os.path.join(path, "info.log"))
    info_fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    info_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
    debug_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s | %(lineno)d:%(funcName)s')

    ch.setFormatter(info_formatter)
    info_fh.setFormatter(info_formatter)
    debug_fh.setFormatter(debug_formatter)

    logger.addHandler(ch)
    logger.addHandler(debug_fh)
    logger.addHandler(info_fh)

    return logger

def train_model(args, builder_class):
    hparams_path = args.hparams

    with open(hparams_path, "r") as f_handle:
        hparams_dict = json.load(f_handle)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    root_dir = os.path.join(hparams_dict["root_dir"], "%s/" % timestamp)

    logger = init_logger(root_dir)
    logger.info("Loaded hyper-parameter configuration from file: %s" % hparams_path)
    logger.info("Hyper-parameters: %s" % str(hparams_dict))
    hparams_dict["root_dir"] = root_dir

    hparams = collections.namedtuple("HParams", sorted(hparams_dict.keys()))(**hparams_dict)

    with open(os.path.join(root_dir, "hparams.json"), "w") as f_handle:
        json.dump(hparams._asdict(), f_handle, indent=2)

    # Build graph
    model = builder_class(hparams)
    if hparams.data_type == "ubuntu":
        model.ubuntu_train(args.pretrained_model)
    elif hparams.data_type == "advising":
        model.advising_train(args.pretrained_model)

def evaluate_model(args, builder_class):
    hparams_path = args.hparams

    with open(hparams_path, "r") as f_handle:
        hparams_dict = json.load(f_handle)

    hparams = collections.namedtuple("HParams", sorted(hparams_dict.keys()))(**hparams_dict)

    # Build graph
    model = builder_class(hparams)
    model.evaluate(args.evaluate)

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser(description="DSTC7 Baseline - Dual Encoder")
    arg_parser.add_argument("--hparams", dest="hparams", required=True,
                            help="Path to the file that contains hyper-parameter settings. (JSON format)")
    arg_parser.add_argument("--model", dest="model",type=str, default=None,
                            help="Path to the file that contains hyper-parameter settings. (JSON format)")
    arg_parser.add_argument("--pretrained_model", dest="pretrained_model", type=str, default=None,
                            help="Path to the pretrained model")
    arg_parser.add_argument("--evaluate", dest="evaluate", type=str, default=None,
                            help="Path to the saved model.")
    args = arg_parser.parse_args()
    model_name = args.model

    if model_name == "dual_encoder_LSTM":
        model = dual_encoder_LSTM.DualEncoderLSTM
    elif model_name == "esim_word":
        model = esim_word.ESIMWord
    elif model_name == "esim_sentence_att":
        model = esim_sentence_att.ESIMSentenceAtt
    elif model_name == "esim_utt_att_glove":
        model = esim_utt_att_glove.ESIMUttAttGlove
    elif model_name == "esim_utt_att_elmo":
        model = esim_utt_att_elmo.ESIMUttAttElmo

    print("The model is %s" % model_name)

    if args.evaluate is not None:
        evaluate_model(args, model)
    else:
        train_model(args, model)