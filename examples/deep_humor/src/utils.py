import configparser
import os
import zipfile

import nltk
import numpy as np
import wget
from sklearn.metrics import accuracy_score, f1_score, recall_score, \
    precision_score

import constants
import hybrid_vector_generator


def calc_metric(y_true, y_pred):
    """
    Calculates accuracy, macro preicision, recall and f1 scores
    :param y_true: Batch of true labels
    :param y_pred: Batch of predicted labels
    :return:
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    return acc, prec, rec, f1


def read_config(config_file):
    """
    Reads the configuration file and creates a dictionary
    :param config_file: Ini file path
    :return:
    """
    config = configparser.ConfigParser()
    config.read(config_file)
    conf_dict = dict()

    conf_dict['lr'] = float(config["MODEL"]['learning rate'])
    conf_dict['dropout'] = float(config["MODEL"]['dropout keep'])
    conf_dict['optimizer'] = config["MODEL"]['optimizer']
    conf_dict['timestep'] = int(config["MODEL"]['timestep'])
    conf_dict['word_vector_dim'] = int(
        config["MODEL"]['word embedding dimension'])
    conf_dict['char_embeddings_dim'] = int(
        config["MODEL"]['character embeddings dimension'])
    conf_dict['lstm_hidden'] = int(config["MODEL"]['lstm hidden state dim'])
    conf_dict['batch_size'] = int(config["MODEL"]['batch size'])
    conf_dict['domain'] = config["GENERAL"]['domain']
    conf_dict['char_max_word'] = int(config["MODEL"]['char max word'])
    conf_dict['early_stopping'] = int(config["MODEL"]['early stopping'])
    conf_dict['train_epochs'] = int(config["GENERAL"]['training epochs'])
    conf_dict['n_classes'] = int(config["MODEL"]['classes'])
    conf_dict['random_seed'] = int(config["MODEL"]['random seed'])
    conf_dict['cross_val_k'] = int(config["GENERAL"]['cross validation k'])

    return conf_dict


def dir_creator(dirs_list):
    """
    Creates directories if they don't exist.
    :param dirs_list: List of absolute directory paths
    :return:
    """
    for d in dirs_list:
        if not os.path.exists(d):
            os.makedirs(d)
            print("Created directory", d)


def extract_zip(file, ext_dir):
    """
    Extracts the zip file to a chosen directroy
    :param file: Zip file path
    :param ext_dir: Extraction directory
    :return:
    """
    print("Extracting", file, "to", ext_dir)
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(ext_dir)
    print("Extraction finished!\n")


def download_data(download_dir="/tmp"):
    """
    Downloads the dataset and resources required for the project.
    NOTE: You should have at least 5GB of disk space available.
    :return:
    """
    glove_dir = os.path.join(constants.RSC, "glove")
    tf_weights = os.path.join(constants.RSC, "tf_weights")

    dir_creator(
        [constants.DATA, constants.RSC, glove_dir, tf_weights])

    # Twitter glove vectors
    glove_name = os.path.join(download_dir, "glove.zip")
    if not os.path.exists(glove_name):
        print("Downloading Twitter Glove vector from", constants.GLOVE_TWITTER)
        print("This may take a while because the file size is 1.4GB")
        wget.download(constants.GLOVE_TWITTER, glove_name)
        print("Downloaded to", glove_name, "\n")
    extract_zip(glove_name, glove_dir)

    # Dataset
    print("Downloading dataset")
    for link, file in zip([constants.TRAIN, constants.VALIDATION,
                           constants.VALIDATION_NO_LABELS],
                          ["train", "valid", "eval"]):
        file = os.path.join(download_dir, file)
        if not os.path.exists(file):
            print("Downloading", link)
            wget.download(link, file)

        extract_zip(file, constants.DATA)

        # TODO Add trained weights download


def int_to_one_hot(input, num_classes):
    x = np.zeros(num_classes)
    x[input] = 1
    return x


def shuffle_data(data):
    """
    Shuffles arbitrary number of data lists
    :param data: List of array which are logically connected and need to be
    shuffled so the order is preserved
    :return: List of shuffled arrays
    """
    indices = np.arange(len(data[0]), dtype=np.int32)
    np.random.shuffle(indices)
    return [x[indices] for x in data]


def split_data(input, labels, test_size=0.4):
    """
    Split the dataset to training, development and test datasets.
    Test set is always half the size of the validation set
    :param chr_embds:
    :param treebank:
    :param pos_tags:
    :param test_size: Validation + test set size
    :return: (train_input, valid_input, test_input, train_labels, valid_labels,
     test_labels)
    """
    # Training set
    ts = int((1 - test_size) * input.shape[0])
    train_word, tmp_word = input[:ts], input[ts:]
    train_label, tmp_label = labels[:ts], labels[ts:]

    # Validation + test set
    tss = int(0.5 * train_word.shape[0])

    valid_word, test_word = tmp_word[:tss], tmp_word[tss:]
    valid_label, test_label = tmp_label[:tss], tmp_label[tss:]

    return train_word, valid_word, test_word, \
           train_label, valid_label, test_label


def extract_data(line):
    """
    Extracts a floating value from the statistical result
    :param line:
    :return:
    """
    data = line.rstrip().split(" ")[-1]
    return float(data[:-1])  # remove percentange


def read_log_file(file):
    """
    Extracts final evaluation metrics from the log files
    :param file: Log file
    :return: Accuracy, precision, recall and f1 arrays
    """
    acc, prec, rec, f1s = [], [], [], []
    with open(file, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "Finished epoch 1" in line:
            data = lines[i - 6:i - 2]
            acc.append(extract_data(data[0]))
            prec.append(extract_data(data[1]))
            rec.append(extract_data(data[2]))
            f1s.append(extract_data(data[3]))

    return np.array(acc), np.array(prec), np.array(rec), np.array(f1s)


def export_twitter_for_fastText(train_dir, export_file):
    """
    Writes twitter files to a txt file
    :param train_dir:
    :param export_file:
    :return:
    """
    with open(export_file, "w") as f:
        for hashtag in os.listdir(train_dir):
            input_filename = os.path.join(train_dir, hashtag)
            tweets = hybrid_vector_generator.load_input_file(
                input_filename)  # (tweet_id, tweet_text,
            # tweet_level)

            filter = {"``", "`", "..", ',', ".", "!", "?", ";", "&", "(", ")", "'", "''",
                      "#", ' \ '":", "...", "-", "+"}

            for tweet_id, tweet_text, tweet_level in tweets:
                tweet = tweet_text.lower()
                for word in tweet.split():
                    if word.startswith('@') or word.startswith(
                            '.@') or word.startswith('http'):
                        tweet = tweet.replace(' ' + word, "")  # if it is on the end
                        tweet = tweet.replace(word + ' ', "")  # if it is on the begining

                tweet = " ".join([x for x in nltk.word_tokenize(tweet) if x not in filter])
                f.write(tweet + "\n")


if __name__ == "__main__":
    train_dir = "/home/bartol/Documents/Work/apt_project/dataset/train_data"
    export = "fasttext.txt"
    export_twitter_for_fastText(train_dir=train_dir, export_file=export)
