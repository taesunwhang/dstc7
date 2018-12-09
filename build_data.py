import os
from data_helpers import *

FILE = "ubuntu"

if FILE == "ubuntu":
    TRAIN_PATH = "data/ubuntu/ubuntu_train_subtask_1.json"
    DEV_PATH = "data/ubuntu/ubuntu_dev_subtask_1.json"
    TEST_PATH = "data/ubuntu/ubuntu_test_subtask_1.json"
    VOCAB_PATH = "data/ubuntu/ubuntu_subtask_1.vocab.txt"

    TRIMMED_PATH = "data/ubuntu_subtask_1_glove_300_dim_trimmed.npz"
    PICKLE_TRAIN_PATH = "data/ubuntu_subtask_1_tokenized.txt"
    PICKLE_DEV_PATH = "data/ubuntu_subtask_1_dev_tokenized.txt"
    CHAR_VOCAB_PATH = "data/ubuntu_subtask_1_char.vocab.txt"

elif FILE == "advising":
    TRAIN_PATH = "data/advising/advising_train_subtask_1.json"
    DEV_PATH = "data/advising/advising_dev_subtask_1.json"
    TEST_PATH = "data/advising/advising_test_subtask_1.json"
    TEST2_PATH = "data/advising/advising_test_subtask_1_case2.json"
    VOCAB_PATH = "data/advising/advising_subtask_1.vocab.txt"

    TRIMMED_PATH = "data/advising/advising_subtask_1_glove_300_dim_trimmed.npz"
    PICKLE_DATA_PATH = "data/advising_scenario_1_tokenized.txt"
    PICKLE_DEV_PATH = "data/advising_scenario_1_dev_tokenized.txt"
    CHAR_VOCAB_PATH = "data/advising.scenario-1_char.vocab.txt"

GLOVE_PATH = "/mnt/raid5/taesun/dstc7/glove.42B.300d.txt"

def build_data():
    test_vocab = get_vocabs(TEST_PATH, "test")
    print("%s vocab is in test data" % len(test_vocab))

    # test2_vocab = get_vocabs(TEST2_PATH, "test")
    # print("%s vocab is in test2 data" % len(test2_vocab))
    #
    dev_vocab = get_vocabs(DEV_PATH, "dev")
    print("%s vocab is in dev_data" % len(dev_vocab))

    train_vocab = get_vocabs(TRAIN_PATH, "train")
    print("%s vocab is in train_data" % len(train_vocab))

    vocab = train_vocab | dev_vocab | test_vocab | test2_vocab
    print("trian, dev, test, test2 : %s vocab" % len(vocab))

    exit()

    # vocab.add("<S>")
    # vocab.add("</S>")
    #
    # write_vocab(vocab, VOCAB_PATH)

    train_vocab, word2id = load_vocab(VOCAB_PATH)
    print(len(train_vocab))

    # if not os.path.exists(CHAR_VOCAB_PATH):
    #     write_char_vocab(train_vocab, CHAR_VOCAB_PATH)

    glove_vocab = load_glove_vocab(GLOVE_PATH)

    vocab = set(train_vocab) & glove_vocab

    print("%s vocab is in Glove" % len(vocab))

    export_trimmed_glove_vectors(word2id,GLOVE_PATH,TRIMMED_PATH,300)

if __name__ == '__main__':
    # make_valid_data(TRAIN_PATH, "./ubuntu_train.txt")
    # make_valid_data(DEV_PATH, "./ubuntu_dev.txt")
    # data_stat("test_%s" % FILE, TEST_PATH)
    build_data()
