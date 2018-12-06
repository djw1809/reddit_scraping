import pandas as pd
import numpy as np
import re
import spacy
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader
import torch
import fastText.FastText as fast
#TODO: Add a preprocessing funcion that formats the strings in the dataframe before we reformat it below
# ''.join(s for s in string if ord(s)>31 and ord(s)<126)
#    string = string.lower()
# string = re.sub(r"([.!?,'/()])", r" \1 ", string)



###########DATA CLEANING/PREPROCESSING ##########################

def clean_comments(dataset):
    '''dataset(pandas dataframe) - data to clean'''
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
