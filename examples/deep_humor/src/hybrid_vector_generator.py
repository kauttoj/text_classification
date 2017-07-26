"""
Generates intermediary pickle files that consist of (word_vector, char_vector,
label) triplets.
"""

from __future__ import print_function, absolute_import, division

import csv
import os
import pickle
import sys

import constants
import dataset_parser
import utils


def generate(train_dir, output_dir, config):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading Glove... Please wait!")
    glove = dataset_parser.loadGlove(constants.GLOVE_PATH)
    print("Glove loaded! Generating vectors...")
    input_files = os.listdir(train_dir)
    target_hashtags = [os.path.splitext(gf)[0] for gf in input_files]
    print('Target hashtags: {} ({})'.format(len(target_hashtags),
                                            ', '.join(target_hashtags)))

    for hashtag in target_hashtags:
        input_filename = os.path.join(train_dir, hashtag + '.tsv')
        output_filename = os.path.join(output_dir, hashtag + '.pickle')
        tweets = load_input_file(
            input_filename)  # (tweet_id, tweet_text, tweet_level)

        data = []
        for tweet_id, tweet_text, tweet_level in tweets:
            word_vector = dataset_parser.createGlovefromTweet(glove, tweet_text,
                                                              config["word_vector_dim"],
                                                              config["timestep"])

            char_vector = dataset_parser.tweet_to_integer_vector(tweet_text,
                                                                 timestep=
                                                                 config['timestep'],
                                                                 max_word_size=
                                                                 config["char_max_word"])

            data.append((word_vector, char_vector, tweet_level))

        write_output_file(output_filename, data)

    print("Stored pickles to: " + output_dir)


def load_input_file(filename):
    tweets_list = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE,
                            escapechar=None)
        for row in reader:
            tweets_list.append(row)
    return tweets_list


def write_output_file(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage:', __file__, '<train_data> <output_dir> <config_file>')
        print('train_data directory contains train tsv files for each theme.')
        print('output directory for storing vector pickles.')
        print('Example call:')
        print(
            'python3 hybrid_vector_generator.py ../dataset/train_data output configs/cnn_lstm.ini')
        sys.exit(1)

    _, train_dir, output_dir, config_file = sys.argv
    generate(train_dir, output_dir, utils.read_config(config_file))
