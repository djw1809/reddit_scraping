import pandas as pd
import numpy as np
import re
import spacy
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader
import torch
import fastText.FastText as fast
#TODO: Add a preprocessing funcion that formats the strings in the dataframe before we reformat it below
# ''.join(s for s in string if ord(s)>31 and ord(s)<126)
#    string = string.lower()
# string = re.sub(r"([.!?,'/()])", r" \1 ", string)



###########DATA CLEANING/PREPROCESSING ##########################
def random_train_test_split(big_label_1, big_label_0, num_samples, train_proportion):
    '''given two datasets where each dataset contains samples with the same label creates random train/test splits
        for each dataset where test_big_label_i + train_big_label_i = num_samples and full_train = train_big_label_1 + train_big_label_0 '''

    label1_set = big_label1.sample(num_samples)
    label0_set = big_label2.sample(num_samples)

    gss = GroupShuffleSplit(n_splits = 1, train_size = .8)

    id01, id02 = next(gss.split(label0_set, groups = label0_set.index))
    id11, id12 = next(gss.split(label1_set, groups = label1_set.index))

    label0_train, label0_test = label0_set.iloc[id01], label0_set.iloc[id02]
    label1_train, label1_test = label1_set.iloc[id11], label1_set.iloc[id12]

    train = label0_train.append(label1_train)
    test = label0_test.append(label0_test)


    return train, test


def clean_comments(dataset):
    '''dataset(pandas dataframe) - data to clean'''

    dataset  = dataset.dropna(how = 'any') 

    for i in dataset.index:
        comment = dataset.loc[i, 'comment body']

        if comment == '[removed]' or comment == '[deleted]':
            dataset = dataset.drop(i)

        else:
            comment = ''.join(s for s in comment if ord(s)>31 and ord(s)<126) #encode everything to ascii and remove special characters
            comment = comment.lower()
            comment = re.sub(r"([.!?,'/()])", r" \1 ", comment)

            dataset.loc[i, 'comment body'] = comment

    dataset.index = range(len(dataset))
    return dataset





def scrape_output_to_model_sets(dataset1, dataset2, labeled = True):
    ''' dataset1 (pandas dataframe) - output of reddit scrape - should represent all same label
        dataset2 (pandas dataframe) - output of reddit scrape - should represent all same label '''
    dataset1.index = range(len(dataset1))
    dataset2.index = range(len(dataset2))

    if labeled:
        output_dataframe1 = pd.DataFrame(np.zeros((len(dataset1.index), 2)), columns = ['label', 'comment'])
        output_dataframe2 = pd.DataFrame(np.zeros((len(dataset2.index), 2)), columns = ['label', 'comment'])

        output_dataframe1['label'] = 1
        output_dataframe2['label'] = 0

        output_dataframe1['comment'] = dataset1['comment body']
        output_dataframe2['comment'] = dataset2['comment body']

        final_output = output_dataframe1.append(output_dataframe2)
    else:
        output_dataframe1 = pd.DataFrame(np.zeros((len(dataset1.index), 1)), columns = ['comment'])
        output_dataframe2 = pd.DataFrame(np.zeros((len(dataset2.index), 1)), columns = ['comment'])

        output_dataframe1['comment'] = dataset1['comment body']
        output_dataframe2['comment'] = dataset2['comment body']

        final_output = output_dataframe1.append(output_dataframe2)
        final_output.index = range(len(final_output))



    return final_output
