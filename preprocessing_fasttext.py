import pandas as pd
import numpy as np
import re
import spacy
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader
#TODO: Add a preprocessing funcion that formats the strings in the dataframe before we reformat it below
# ''.join(s for s in string if ord(s)>31 and ord(s)<126)
#    string = string.lower()
# string = re.sub(r"([.!?,'/()])", r" \1 ", string)

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

        output_dataframe1['label'] = '__label__1'
        output_dataframe2['label'] = '__label__2'

        output_dataframe1['comment'] = dataset1['comment body']
        output_dataframe2['comment'] = dataset2['comment body']

        final_output = output_dataframe1.append(output_dataframe2)
    else:
        output_dataframe1 = pd.DataFrame(np.zeros((len(dataset1.index), 1)), columns = ['comment'])
        output_dataframe2 = pd.DataFrame(np.zeros((len(dataset2.index), 1)), columns = ['comment'])

        output_dataframe1['comment'] = dataset1['comment body']
        output_dataframe2['comment'] = dataset2['comment body']

        final_output = output_dataframe1.append(output_dataframe2)




    return final_output

def encode_corpus_one_hot_stupid(dataset):
    corpus = ''

    for i in len(dataset.index):
        corpus += dataset.loc[i, 'comment']

    nlp = spacy.load('en')
    tokenized_corpus = nlp.tokenizer(corpus)
    list_of_tokens = [tok.text for tok in tokenized_corpus]
    list_of_tokens = list(set(list_of_tokens))

    vector_dict = {}
    vectors = np.zeros(len(list_of_tokens))

    for i in range(len(list_of_tokens)):
        vectors[i,i] = 1
        vector_dict[list_of_tokens[i]] = vectors[i, :]

    return vectors, list_of_vectors

def encode_corpus_one_hot_smart(dataset, corpus = False):

    if corpus == False:
        corpus = ''

        for i in range(len(dataset.index)):
            corpus += dataset.loc[i, 'comment']
    else:
        corpus = dataset

    nlp = spacy.load('en')
    tokenized_corpus = nlp.tokenizer(corpus)
    list_of_tokens = [tok.text for tok in tokenized_corpus]
    list_of_tokens = list(set(list_of_tokens))

    X = np.asarray(list_of_tokens).reshape(-1,1)
    encoder = OneHotEncoder()

    encoder.fit(X)

    return encoder

class corpus_to_one_hot(Dataset):

    def __init__(self, data):
        self.encoding = encode_corpus_one_hot_smart

    def __getitem__(self, token):
        dummy1 = []
        dummy2 = []
        dummy2.append(token)
        dummy1.append(dummy2)
        return torch.Tensor(self.enoding.transform(dummy1).toarray())

    def inverse_lookup(self, vector):
        vector = np.asarray(vector)
        return self.encoding.inverse_transform(vector)



##using fasttext with output:
## training_word_embeddings: ./fasttext skipgram -input *corpus cleaned by this module* -output *where you want model parameters saved* (-minn * * -maxn * * -dim * *)
## nearest neighbor queries: ./fastteext nn *location of word embedding*.bin
