import os

ROOT = os.path.dirname(__file__)

RSC = os.path.join(ROOT, "resources")

DATA = os.path.join(ROOT, "data")

TEMP_OUTPUT = os.path.join(ROOT, "temp-output")

GLOVE_PATH = os.path.join(ROOT, "./resources/glove/glove.twitter.27B.100d.txt")

# Data and resources URLs
GLOVE_TWITTER = "http://nlp.stanford.edu/data/glove.twitter.27B.zip"
TRAIN = "http://alt.qcri.org/semeval2017/task6/data/uploads/train_data.zip"
VALIDATION = "http://alt.qcri.org/semeval2017/task6/data/uploads/gold_data.zip"
VALIDATION_NO_LABELS = "http://alt.qcri.org/semeval2017/task6/data/uploads/evaluation_dir.zip"


TF_WEIGHTS = os.path.join(RSC, "weights")
LOGS = os.path.join(RSC, "logs")

CONFIGS = os.path.join(ROOT, "configs")

letter_to_int_dict = {
    'padding': 0,
    'a': 1,
    'b': 2,
    'c': 3,
    'd': 4,
    'e': 5,
    'f': 6,
    'g': 7,
    'h': 8,
    'i': 9,
    'j': 10,
    'k': 11,
    'l': 12,
    'm': 13,
    'n': 14,
    'o': 15,
    'p': 16,
    'r': 17,
    's': 18,
    't': 19,
    'u': 20,
    'v': 21,
    'z': 22,
    'x': 23,
    'y': 24,
    'w': 25,
    'q': 26,
    '-': 27,
    '.': 28,
    ',': 29,
    '!': 30,
    '?': 31,
    '_': 32,
    '$': 33,
    '&': 34,
    ')': 35,
    '(': 36,
    '+': 37,
    '"': 38,
    "'": 39,
    " ": 40
}


def map_letter_to_int(value):
    """
    Maps given letter or string to integer. For non-important characters it will return 42.
    :param value: String or character to map. Possbile strings: 'end' and 'padding'.
    :return: Integer value for given string or character.
    """
    return letter_to_int_dict.get(value,
                                  41)  # 42 is used for non-important characters
