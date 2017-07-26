#!/usr/bin/env python

from __future__ import print_function, absolute_import, division
import sys
import os
import csv
import itertools
import math

def evaluate(submission_dir, gold_dir):
    gold_files = os.listdir(gold_dir)
    target_hashtags = [os.path.splitext(gf)[0] for gf in gold_files]
    print('Target hashtags: {} ({})'.format(len(target_hashtags), ', '.join(target_hashtags)))

    print('Submission files: {}'.format(len(os.listdir(submission_dir))))

    total_tweets = 0
    distances = []
    for hashtag in target_hashtags:
        gold_filename = os.path.join(gold_dir, hashtag + '.tsv')
        gold_dict = load_gold_file(gold_filename)

        pred_filename = os.path.join(submission_dir, hashtag + '_PREDICT.tsv')
        predictions = load_predictions(pred_filename)

        if len(predictions) != len(gold_dict):
            print('Warning! Incorrect number of total predictions for the hashtag {} - {}/{}'.format(
                hashtag, len(predictions), len(gold_dict)))

        cur_moves = 0
        cur_correct = 0
        for tweet, gold_label in gold_dict.items():
            if tweet not in predictions:
                cur_moves += abs(gold_label - 2)
                continue

            diff = abs(predictions[tweet] - gold_label)
            cur_moves += diff
            if diff == 0:
                cur_correct += 1

        cur_distance = len(gold_dict) * cur_moves / 22
        distances.append(cur_distance)
        total_tweets += len(gold_dict)

        print('Hashtag: {} - {:.3f}, distance: {}, moves: {}, correct: {}, total: {}'.format(
            hashtag, cur_distance / len(gold_dict), cur_distance, cur_moves, cur_correct, len(gold_dict)))

    save_scores(distances, total_tweets)


def nCr(n, r):
    '''http://stackoverflow.com/a/4941846'''
    f = math.factorial
    return f(n) / f(r) / f(n - r)


def load_gold_file(filename):
    gold_dict = {}

    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar=None)
        for row in reader:
            tweet_id, tweet_text, tweet_label = row
            gold_dict[tweet_id] = int(tweet_label)

    return gold_dict


def load_predictions(filename):
    predictions = {}
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar=None)
        for i, row in enumerate(reader):
            bucket = 2
            if i > 0:
                bucket = 1
            if i > 9:
                bucket = 0

            tweet = row[0]
            predictions[tweet] = bucket

    return predictions


def save_scores(distances, total_tweets):
    dist = sum(distances) / total_tweets
    print('Final distance: {:.3f} ({})'.format(dist, dist))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage:', __file__, '<prediction_dir> <gold_dir>')
        print(' ',
              'The files in the <prediction_dir> should be named like their corresponding data files, but with _PREDICT prior to the .tsv extension')
        print('  ', 'For example, Hahstag_File.tsv should be named Hashtag_File_PREDICT.tsv')
        print(' ',
              'The files in the <prediction_dir> should contain tweet ids ranked in decreasing order of funniness as follows:')
        print('  ', '<winning tweet_id>')
        print('  ', '<top10 but not winning tweet_id>')
        print('  ', '...')
        print('  ', '<top10 but not winning tweet_id>')
        print('  ', '<not in top10 tweet_id>')
        print('  ', '...')
        print('  ', '<not in top10 tweet_id>')
        print(' ',
              'The files in the <gold_dir> should be files formatted as have been released in train/trail data files')
        sys.exit(1)

    # as per the metadata file, input and output directories are the arguments
    _, submission_dir, gold_dir = sys.argv
    evaluate(submission_dir, gold_dir)