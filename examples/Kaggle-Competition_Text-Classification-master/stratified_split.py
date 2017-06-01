
''' Function to implement Stratified splitting method to split the dataset into training and te    sting dataset.
    Input: y is the label of the whole dataset, proportion is the percent of training dataset
    Output: the indices of the training dataset, and the indices of the test dataset '''

import numpy as np  

def stratified_split(y,train_proportion):
    '''Generates indices, making random stratified split into training set and testing sets
    with proportions train_proportion and (1-train_proportion) of initial sample.
    y is any iterable indicating classes of each observation in the sample.
    Initial proportions of classes inside training and 
    testing sets are preserved (stratified sampling).
    '''

    # y=np.array(y)
    train_inds = np.zeros(len(y),dtype=bool)
    test_inds = np.zeros(len(y),dtype=bool)
    values = np.unique(y)
    for value in values:
        value_inds = np.nonzero(y==value)[0]
        np.random.shuffle(value_inds)
        n = int(train_proportion*len(value_inds))

        train_inds[value_inds[:n]]=True
        test_inds[value_inds[n:]]=True
 
    return train_inds,test_inds


