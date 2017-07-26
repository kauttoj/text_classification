#!/usr/bin/env python
from __future__ import print_function, absolute_import, division

import csv
import os
import sys

import numpy as np

import constants
import dataset_parser
import model_evaluation
import utils


def generate(input_dir, output_dir, model, config):
    print("Loading glove")
    glove = dataset_parser.loadGlove(constants.GLOVE_PATH)
    print("loaded glove file")

    input_files = os.listdir(input_dir)
    target_hashtags = [os.path.splitext(gf)[0] for gf in input_files]
    print('Target hashtags: {} ({})'.format(len(target_hashtags),
                                            ', '.join(target_hashtags)))

    for hashtag in target_hashtags:
        input_filename = os.path.join(input_dir, hashtag + '.tsv')
        output_filename = os.path.join(output_dir, hashtag + '_PREDICT.tsv')
        tweets = load_input_file(input_filename)  # (tweetID, tweetText)

        results = {}
        # make tweet combinations and get result
        index = 1
        word_data, char_data = [], []
        for tweetID1, tweet_text1 in tweets:
            results[tweetID1] = 0
            for tweetID2, tweet_text2 in tweets[index:]:
                if tweetID1 == tweetID2:
                    continue

                word_vect1, char_vect1 = get_feature_vector(glove, tweet_text1, config)
                word_vect2, char_vect2 = get_feature_vector(glove, tweet_text2, config)
                word_merged = np.array([word_vect1, word_vect2], dtype=np.float32)
                char_merged = np.array([char_vect1, char_vect2], dtype=np.float32)

                word_data.append(word_merged)
                char_data.append(char_merged)
            index += 1

        network_results = get_classification(model,
                                             np.array(word_data), np.array(char_data))
        index = 1
        pred_index = 0
        for tweetID1, tweet_text1 in tweets:
            for tweetID2, tweet_text2 in tweets[index:]:
                if tweetID1 == tweetID2:
                    continue
                network_result = network_results[pred_index]
                if network_result == 1:
                    increase_counter(results, tweetID1)
                else:
                    increase_counter(results, tweetID2)
                pred_index += 1

            index += 1

        results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        if results[0][0] == results[1][0]:
            print("They are the same!!! NOOOOOOO!")
        write_output_file(output_filename, results)


def increase_counter(dictionary, key):
    if key in dictionary:
        dictionary[key] += 1
    else:
        dictionary[key] = 1


def get_feature_vector(embed_dict, tweet_text, config):
    return (dataset_parser.createGlovefromTweet(embed_dict, tweet_text,
                                                timestep=config['timestep']),
            dataset_parser.tweet_to_integer_vector(tweet_text,
                                                   timestep=config['timestep'],
                                                   max_word_size=config['char_max_word']))


def get_classification(model, word_merged, char_merged):
    return model.predict(word_merged, char_merged, batch_size=128)


def load_input_file(filename):
    tweets_list = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE,
                            escapechar=None)
        for row in reader:
            tweets_list.append(row)
    return tweets_list


def write_output_file(filename, results):
    with open(filename, 'w') as f:
        for tweetID, count in results:
            f.write(tweetID + "\n")


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Usage:', __file__, '<input_dir> <output_dir> <model_path> <config_path>')
        print('Input directory contains tsv files for each theme.')
        sys.exit(1)

    _, input_dir, output_dir, model_path, config_path = sys.argv
    model = model_evaluation.ModelEvaluator(model_path)
    config = utils.read_config(config_path)

    generate(input_dir, output_dir, model, config)
