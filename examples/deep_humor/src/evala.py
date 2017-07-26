#!/usr/bin/env python

from __future__ import print_function, absolute_import, division

import csv
import itertools
import math
import os
import sys


def evaluate(submission_dir, gold_dir):
    gold_files = os.listdir(gold_dir)
    target_hashtags = [os.path.splitext(gf)[0] for gf in gold_files]
    print('Target hashtags: {} ({})'.format(len(target_hashtags),
                                            ', '.join(target_hashtags)))

    print('Submission files: {}'.format(len(os.listdir(submission_dir))))

    total = 0
    correct = 0
    checked_pairs = {}
    for hashtag in target_hashtags:
        gold_filename = os.path.join(gold_dir, hashtag + '.tsv')
        win_list, top10_list, nontop10_list = load_gold_file(gold_filename)

        nb_all_tweets = len(win_list) + len(top10_list) + len(nontop10_list)
        nb_combs = int(round(nCr(nb_all_tweets, 2)))

        cur_total = (1 * 9) + (1 * len(nontop10_list)) + 9 * len(nontop10_list)
        cur_correct = 0
        gold_dict = create_gold_dict(win_list, top10_list, nontop10_list)

        pred_filename = os.path.join(submission_dir, hashtag + '_PREDICT.tsv')
        predictions = load_predictions(pred_filename)

        if len(predictions) != nb_combs:
            print(
                'Warning! Incorrect number of total predictions for the hashtag {} - {}/{}'.format(
                    hashtag, len(predictions), nb_combs))

        for tweet1, tweet2, label in predictions:
            key = (tweet1, tweet2)
            if key in gold_dict and key not in checked_pairs:
                checked_pairs[(tweet1, tweet2)] = True
                checked_pairs[(tweet2, tweet1)] = True

                gold_label = gold_dict[key]
                if label == gold_label:
                    cur_correct += 1

        print('Hashtag: {} - {:.3f}, lines: {}, correct: {}, total: {}'.format(
            hashtag, cur_correct / cur_total, len(predictions), cur_correct,
            cur_total))

        total += cur_total
        correct += cur_correct

    calc_scores(correct, total)


def nCr(n, r):
    '''http://stackoverflow.com/a/4941846'''
    f = math.factorial
    return f(n) / f(r) / f(n - r)


def load_gold_file(filename):
    win_list = []
    top10_list = []
    nontop10_list = []

    label_list_dict = {
        '0': nontop10_list,
        '1': top10_list,
        '2': win_list,
    }

    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE,
                            escapechar=None)
        for row in reader:
            tweet_id, tweet_text, tweet_label = row
            label_list_dict[tweet_label].append(tweet_id)

    assert len(win_list) == 1, 'Win list contains wrong number of items'
    assert len(top10_list) == 9, 'Top10 list contains wrong number of items'

    return win_list, top10_list, nontop10_list


def create_gold_dict(win_list, top10_list, nontop10_list):
    # tweets in the first list are always funnier than in the second
    tweet_lists = [
        (win_list, top10_list),
        (win_list, nontop10_list),
        (top10_list, nontop10_list),
    ]

    gold_dict = {}
    for list1, list2 in tweet_lists:
        for t1, t2 in itertools.product(list1, list2):
            gold_dict[(t1, t2)] = 1
            gold_dict[(t2, t1)] = 0

    assert len(gold_dict) == 2 * (
        (1 * 9) + (1 * len(nontop10_list)) + 9 * len(
            nontop10_list)), 'Incorrect length of the gold dict'

    return gold_dict


def load_predictions(filename):
    predictions = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE,
                            escapechar=None)
        for row in reader:
            tweet1, tweet2, label = row
            predictions.append((tweet1, tweet2, int(label)))

    return predictions


def calc_scores(correct, total):
    accuracy = correct / total
    print('Final accuracy: {:.3f} ({})'.format(accuracy, accuracy))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage:', __file__, '<prediction_dir> <gold_dir>')
        print(' ',
              'The files in the <prediction_dir> should be named like their corresponding data files, but with _PREDICT prior to the .tsv extension')
        print('  ',
              'For example, Hahstag_File.tsv should be named Hashtag_File_PREDICT.tsv')
        print(' ',
              'The files in the <prediction_dir> should be formatted as follows: ')
        print('  ', '<tweet1_id>\\t<tweet2_id>\\t<prediction>')
        print('  ',
              'where <prediction> is 1 if the first tweets is funnier and 0 otherwise')
        print(' ',
              'The files in the <gold_dir> should be files formatted as have been released in train/trail data files')
        sys.exit(1)

    # as per the metadata file, input and output directories are the arguments
    _, submission_dir, gold_dir = sys.argv
    evaluate(submission_dir, gold_dir)
