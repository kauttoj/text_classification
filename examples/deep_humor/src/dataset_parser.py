import logging
import os
import pickle
import re

import nltk
import numpy as np

import constants


def clear_tweet(tweet):
    """
    Clear given tweet from hashtags, mails, links...
    :param tweet: Tweet
    :return: cleared tweet
    """
    tweet = tweet
    for word in tweet.split():
        if word.startswith('@') or word.startswith(
                '.@') or word.startswith('http') or word.startswith('#'):
            tweet = tweet.replace(' ' + word, "")  # if it is on the end
            tweet = tweet.replace(word + ' ', "")  # if it is on the begining
    return tweet


def tweet_to_integer_vector(tweet, max_word_size, timestep):
    """
    Creates characters mappings in the following manner:
    1) Map each character to its id (using a hash map)
    2) Split the sentence into words
    3) For each word, map each char to its ID, and pad for the max_word_size
    4) Do this for every word in the sentence
    5) Creats a sentence representation of dimension max_word_size * timestep

    :param timestep: Sentence timestep
    :param max_word_size: Maximum word size in number of characters
    :param tweet: Tweet in a string form
    :return: Integer vector for given tweet
    """
    tweet = clear_tweet(tweet.lower())

    new_sent = np.zeros(timestep * max_word_size, dtype=np.int32)
    new_sent_temp = []
    for token in tweet.split(" "):

        # Zeroes are used as a padding
        char_embeddings = np.zeros(max_word_size, dtype=np.uint8)
        char_mapped = np.array([constants.map_letter_to_int(c) for c in token])

        if len(char_mapped) > max_word_size:
            char_embeddings = char_mapped[:max_word_size]
        elif len(char_mapped) < max_word_size:
            pad_size = (max_word_size - len(char_mapped)) // 2
            for i in range(len(char_mapped)):
                char_embeddings[i + pad_size] = char_mapped[i]
        else:
            char_embeddings = char_mapped
        new_sent_temp.extend(char_embeddings)

    new_sent_temp = np.array(new_sent_temp)
    if len(new_sent_temp) > timestep * max_word_size:
        new_sent = new_sent_temp[:len(new_sent)]

    elif len(new_sent_temp) < timestep * max_word_size:
        new_sent[:len(new_sent_temp)] = new_sent_temp
    else:
        new_sent = new_sent_temp

    assert new_sent.shape[0] == timestep * max_word_size

    return new_sent


def loadGlove(glove_file):
    embed_dict = {}

    logging.info("Loading glove file...")
    with open(glove_file) as f:
        for line in f:
            split = line.split()
            token = split[0]
            vec = np.array([np.float(x) for x in split[1:]])
            embed_dict[token] = vec

    return embed_dict


def load_fastText_dict(fast_text):
    embed_dict = {}
    logging.info("Loading glove file...")

    with open(fast_text, "r") as f:
        for line in f:
            split = line.split()
            token = split[0]
            vec = np.array([np.float(x) for x in split[1:]])
            embed_dict[token] = vec

    return embed_dict


def read_file_by_line_and_tokenize(file_path):
    """
    Reads the twee document file and tokenizes it.
    :param file_path:
    :return: List of tokenized tweets
    """
    tweets = [line.rstrip('\n') for line in
              open(file_path, 'r', encoding="utf8")]
    return [re.split('\t', tweet) for tweet in tweets]


def prepare_dataset_for_taskB(glove_file,
                              data_path,
                              pickleInputDir,
                              pickleLabelDir,
                              embedding_dim=100,
                              timestep=25):
    """
    Creates per token tweet embeddings using the Glove 100-D vectors.
    Exports embeddings for tweets and labels to a pickle file.

    Labels: [0,0,1] = 2   [0,1,0] = 1  [1,0,0] = 0

    :param embedding_dim: Word embedding dimension. 100 for Glove vectors
    :param timestep: Maximum sentence length
    :param glove_file: Glove file path
    :param data_path: Files directory path
    :param pickleInputDir: Export pickle directory for inputs
    :param pickleLabelDir: Export pickle directory for inputs
    """

    embed_dict = {}

    logging.info("Loading glove file...")
    with open(glove_file) as f:
        for line in f:
            split = line.split()
            token = split[0]
            vec = np.array([np.float(x) for x in split[1:]])
            embed_dict[token] = vec

    inputMatrix = []
    labelMatrix = []

    for i, filename in enumerate(os.listdir(data_path)):
        print("Parsing file number:", i)

        if filename.endswith(".tsv"):
            tweetsMatrix = []
            current_rows = read_file_by_line_and_tokenize(
                os.path.join(data_path, filename))
            ranks = [word[2] for word in current_rows]
            tokenized = [nltk.word_tokenize(word[1]) for word in current_rows]
            tokenizedCopy = []
            for token in tokenized:
                tokenizedCopy.append(
                    [word.lower() for word in token if word.isalpha()])
            for k, tweet in enumerate(tokenized):
                sentenceRow = np.zeros((embedding_dim, timestep))
                label = np.zeros(3)
                label[int(ranks[k])] = 1
                for j, token in enumerate(tweet[:timestep]):
                    if token in embed_dict:
                        sentenceRow[:, j] = embed_dict[token.lower()]

                inputMatrix.append(sentenceRow)
                labelMatrix.append(label)

    with open(pickleInputDir, "wb") as f:
        pickle.dump(inputMatrix, f)

    with open(pickleLabelDir, "wb") as f:
        pickle.dump(labelMatrix, f)


def createGlovefromTweet(embed_dict, tweetText, embedding_dim=100,
                         timestep=25):
    """
    Method takes tweet and converts it in Glove embedding

    :param embed_dict: Glove dictionary
    :param timestep: Maximum sentence length
    :param embedding_dim: Word embedding dimension. 100 for Glove vectors
    :param tweetText: Tweeter text

    """
    tokens = nltk.word_tokenize(tweetText)
    tokens = [word.lower() for word in tokens]
    sentenceRow = np.zeros((embedding_dim, timestep))
    for j, token in enumerate(tokens[:timestep]):
        if token in embed_dict:
            sentenceRow[:, j] = embed_dict[token]
        else:
            sentenceRow[:, j] = np.ones(embedding_dim) / 20

    return sentenceRow


def create_pair_combs(lst):
    """
    Create all pair combinations from tweets of different humor ranking.
    :param lst:
    :return: List of docuement pairs and the list of label pairs
    """
    index = 1
    pairs_words = []
    pairs_labels = []
    pairs_chrs = []

    for element1 in lst:
        for element2 in lst[index:]:
            if int(element1[-1]) == int(element2[-1]):
                continue
            if int(element1[-1]) < int(element2[-1]):
                # Second pair is funnier
                if np.random.random() > 0.5:
                    concatMatrix = np.array([element1[0], element2[0]], dtype=np.float32)
                    chr_merged = np.array([element1[1], element2[1]], dtype=np.int32)
                    pairs_labels.append(0)
                else:
                    concatMatrix = np.array([element2[0], element1[0]], dtype=np.float32)
                    chr_merged = np.array([element2[1], element1[1]], dtype=np.int32)
                    pairs_labels.append(1)
            else:
                # First pair is funnier
                if np.random.random() > 0.5:
                    concatMatrix = np.array([element2[0], element1[0]], dtype=np.float32)
                    chr_merged = np.array([element2[1], element1[1]], dtype=np.int32)
                    pairs_labels.append(0)
                else:
                    concatMatrix = np.array([element1[0], element2[0]], dtype=np.float32)
                    chr_merged = np.array([element1[1], element2[1]], dtype=np.int32)
                    pairs_labels.append(1)

            # Add word and char level information
            pairs_words.append(concatMatrix)
            pairs_chrs.append(chr_merged)

        index += 1
    return pairs_words, pairs_chrs, pairs_labels


def parse_data(glove_file,
               data_path,
               pickleDir,
               embedding_dim=100,
               timestep=25):
    """
    Creates per token tweet embeddings using the Glove 100-D vectors.
    Exports embeddings to a pickle file.

    :param embedding_dim: Word embedding dimension. 100 for Glove vectors
    :param timestep: Maximum sentence length
    :param glove_file: Glove file path
    :param data_path: Files directory path
    :param pickleDir: Export pickle directory
    """
    embed_dict = {}

    logging.info("Loading glove file...")
    with open(glove_file) as f:
        for line in f:
            split = line.split()
            token = split[0]
            vec = np.array([np.float(x) for x in split[1:]])
            embed_dict[token] = vec

    topicsMatrix = []

    for i, filename in enumerate(os.listdir(data_path)):
        print("Parsing file number:", i)

        if filename.endswith(".tsv"):
            tweetsMatrix = []
            current_rows = read_file_by_line_and_tokenize(
                os.path.join(data_path, filename))
            ranks = [word[2] for word in current_rows]
            tokenized = [nltk.word_tokenize(word[1]) for word in current_rows]
            tokenizedCopy = []
            for token in tokenized:
                tokenizedCopy.append(
                    [word.lower() for word in token if word.isalpha()])
            for k, tweet in enumerate(tokenized):
                sentenceRow = np.zeros((embedding_dim, timestep))
                for j, token in enumerate(tweet[:timestep]):
                    if token in embed_dict:
                        sentenceRow[:, j] = embed_dict[token.lower()]

                tweetsMatrix.append((sentenceRow, ranks[k]))
            topicsMatrix.append(tweetsMatrix)

    with open(pickleDir, "wb") as f:
        pickle.dump(topicsMatrix, f)
