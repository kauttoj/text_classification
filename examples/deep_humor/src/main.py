import argparse
import datetime
import logging
import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

import constants
import dataset_parser
import models
import utils


def create_data_pairs(topics_matrix):
    """
    Create data pair combinations
    """
    X_words, X_chars, y = [], [], []
    for topic in topics_matrix:
        comb_m, comb_chr, comb_l = dataset_parser.create_pair_combs(topic)
        X_words.extend(comb_m)
        X_chars.extend(comb_chr)
        y.extend(
            [utils.int_to_one_hot(x, config['n_classes']) for x in comb_l])
    return np.array(X_words, dtype=np.float32), \
           np.array(X_chars, dtype=np.int32), \
           np.array(y, dtype=np.uint8)


def create_data_sets(pickle_dir):
    """
    Loads, splits and mergers the training and development datasets from
    the pickle files. Processed word_vector, char_vector and label tuples are
    stored into two seperate lists, one for training, and the second one for
    development

    :param train_size: Percentage of train set size (in terms of number of
     documents)
    :param pickle_dir: Training data pickle dir
    :return List of training data, List of development data tuple
    """
    files = os.listdir(pickle_dir)
    data = []
    for file in files:
        file_path = os.path.join(pickle_dir, file)
        with open(file_path, "rb") as f:
            data.append(pickle.load(f))
    return np.array(data)


def create_seperate_dataset(data, num_classes):
    """
    Creates dataset in a form: sentence_repr -> one_hot_humor_label
    Shuffles the data before returning.
    :param data: Train or dev data
    :return: Shuffled word, char and labels
    """
    word_repr, char_repr, labels = [], [], []
    for doc in data:
        for sent in doc:
            word_repr.append(sent[0])
            char_repr.append(sent[1])
            labels.append(utils.int_to_one_hot(int(sent[2]), num_classes))

    word_repr, char_repr, labels = np.array(word_repr), \
                                   np.array(char_repr, dtype=np.int32), \
                                   np.array(labels)

    return utils.shuffle_data([word_repr, char_repr, labels])


def main(config):
    """
    :param final_eval: Whether to do final model training on all data
    :param config: Loaded configuration dictionary
    :return:
    """
    logging.info(
        "Running K-Fold cross validation with k={}".format(
            config['cross_val_k']))

    train_pickle_dir = os.path.join("data", "pickled", "pickled_train")
    if not os.path.exists(train_pickle_dir):
        print("Run `hybrid_vector_generator` script to generate train pickles.")
        return

    # Generate pairs
    all_data = create_data_sets(train_pickle_dir)

    # Create K-fold object generator
    k_fold = KFold(n_splits=config['cross_val_k'], shuffle=True,
                   random_state=config['random_seed'])
    k_fold.get_n_splits(all_data)

    # Do training for every fold
    for fold, (train_index, test_index) in enumerate(
            k_fold.split(all_data)):
        logging.info(
            "Now starting fold {}/{}".format(fold + 1,
                                             config['cross_val_k']))
        train_data = all_data[train_index]
        dev_data = all_data[test_index]

        x_train_word, x_train_chr, y_train = create_data_pairs(train_data)
        x_dev_word, x_dev_chr, y_dev = create_data_pairs(dev_data)

        # Memory cleanup
        del train_data
        del dev_data

        assert x_train_word.shape[3] == config['timestep']
        assert y_train.shape[1] == config['n_classes']
        assert x_train_chr.shape[2] == config['char_max_word'] * config['timestep']
        assert x_train_chr.shape[0] == y_train.shape[0] == x_train_word.shape[0]

        # Mock data
        config['train_examples'] = x_train_word.shape[0]
        config['validation_examples'] = x_dev_word.shape[0]
        config['save_dir'] = os.path.join(constants.TF_WEIGHTS)

        config['train_word'] = None
        config['valid_word'] = None
        config['train_chr'] = None
        config['valid_chr'] = None
        config['train_label'] = None
        config['valid_label'] = None
        # Log configuration
        logging.info("CONFIG:")
        logging.info(
            "\n".join([k + ": " + str(v) for k, v in config.items()]))

        # Add datasets to config
        config['train_word'] = x_train_word
        config['valid_word'] = x_dev_word
        config['train_chr'] = x_train_chr
        config['valid_chr'] = x_dev_chr
        config['train_label'] = y_train
        config['valid_label'] = y_dev

        # +1 for unknown words
        config['char_vocab_size'] = len(constants.letter_to_int_dict) + 1

        # Train all three
        net = models.BILSTM_FC(config)
        net.train()

        # Recreate graph
        tf.reset_default_graph()


def parse_arguments():
    """
    Parses commmand line arguments
    :return: Parseed objects
    """
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Configuration file path. They are defined in the '
                             'src/configs folder')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    config = utils.read_config(args.config)

    seed = config['random_seed']
    np.random.seed(seed)

    # Setup logging
    utils.dir_creator([constants.LOGS, constants.TF_WEIGHTS])

    log_name = config['domain'] + "_" + str(
        datetime.datetime.now().strftime("%d_%m_%Y_%H:%M")) + ".log"
    log_file = os.path.join(constants.LOGS, log_name)
    print("Logging to", log_file)
    logging.basicConfig(filename=log_file,
                        filemode='w',
                        format='%(asctime)s: %(levelname)s: %(message)s',
                        level=logging.DEBUG, datefmt='%d/%m/%Y %I:%M:%S %p')

    logging.info("Numpy random seed set to " + str(seed))
    main(config=config)
