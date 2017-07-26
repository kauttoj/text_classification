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
    print("Loading glove file...")
    glove = dataset_parser.loadGlove(constants.GLOVE_PATH)
    print("Loaded glove file!")

    input_files = os.listdir(input_dir)
    target_hashtags = [os.path.splitext(gf)[0] for gf in input_files]
    print('Target hashtags: {} ({})'.format(len(target_hashtags),
                                            ', '.join(target_hashtags)))

    for hashtag in target_hashtags:
        input_filename = os.path.join(input_dir, hashtag + '.tsv')
        output_filename = os.path.join(output_dir, hashtag + '_PREDICT.tsv')
        tweets = load_input_file(input_filename)  # (tweetID, tweetText)

        results = []
        # make tweet combinations and get result
        index = 1
        word_data = []
        char_data = []
        for tweetID1, tweet_text1 in tweets:
            for tweetID2, tweet_text2 in tweets[index:]:
                if tweetID1 == tweetID2:
                    continue

                word_vect1, char_vect1 = get_feature_vector(glove, tweet_text1, config)
                word_vect2, char_vect2 = get_feature_vector(glove, tweet_text2, config)
                word_merged = np.array([word_vect1, word_vect2])
                char_merged = np.array([char_vect1, char_vect2])
                word_data.append(word_merged)
                char_data.append(char_merged)

            index += 1

        print("Predicting...")
        predict_data = model.predict(np.array(word_data), np.array(char_data),
                                     batch_size=256)

        index = 1
        pred_index = 0
        for tweetID1, tweet_text1 in tweets:
            for tweetID2, tweet_text2 in tweets[index:]:
                if tweetID1 == tweetID2:
                    continue
                results.append((tweetID1, tweetID2, predict_data[pred_index]))
                pred_index += 1
            index += 1

        write_output_file(output_filename, results)


def get_feature_vector(embed_dict, tweet_text, config):
    return (dataset_parser.createGlovefromTweet(embed_dict, tweet_text,
                                                timestep=config['timestep']),
            dataset_parser.tweet_to_integer_vector(tweet_text,
                                                   timestep=config[
                                                       'timestep'],
                                                   max_word_size=config['char_max_word']))


def get_classification(model, word_merged, char_merged):
    return model.predict(word_merged, char_merged)


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
        for tweetID1, tweetID2, result in results:
            f.write(tweetID1 + "\t" + tweetID2 + "\t" + str(result) + "\n")


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Usage:', __file__, '<input_dir> <output_dir> <model_path> <config_path>')
        print('Input directory contains tsv files for each theme.')
        sys.exit(1)

    _, input_dir, output_dir, model_path, config_path = sys.argv
    model = model_evaluation.ModelEvaluator(model_path)
    config = utils.read_config(config_path)

    generate(input_dir, output_dir, model, config)
