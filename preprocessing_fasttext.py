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

#################ONE-HOT ENCODING######################

def encode_corpus_one_hot_stupid(dataset):
    corpus = ''

    for i in len(dataset.index):
        corpus += ' ' + dataset.loc[i, 'comment']

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
    '''returns an sklearn one-hot encoding of a corpus - basically a rougher version of corpus_To_one_hot'''

    if corpus == False:
        corpus = ''

        for i in range(len(dataset.index)):
            corpus += ' '+ dataset.loc[i, 'comment']
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


def create_comment_vector(comment, one_hot_encoding, nlp):
    '''creates a comment vector via a bag of words average of word vectors in one_hot_encoding'''
    tokenized_comment_object = nlp.tokenizer(comment)
    tokenized_comment = [tok.text for tok in tokenized_comment_object]

    vector = np.zeros((1, one_hot_encoding.dimension()))

    for i in range(len(tokenized_comment)):
        word_vector = one_hot_encoding[tokenized_comment[i]]
        vector += word_vector

    vector = vector / len(tokenized_comment)

    return vector


class corpus_to_one_hot(Dataset):
    '''computes word vectors for each word in data via a one-hot encoding'''
    def __init__(self, data):
        self.encoding = encode_corpus_one_hot_smart(data, corpus = False)
        self.data = data
        self.nlp = spacy.load('en')

    def dimension(self):
        return self.encoding.transform([['.']]).toarray().shape[1]

    def __getitem__(self, token):
        dummy1 = []
        dummy2 = []
        dummy2.append(token)
        dummy1.append(dummy2)
        return torch.Tensor(self.encoding.transform(dummy1).toarray())

    def inverse_lookup(self, vector):
        vector = np.asarray(vector)
        return self.encoding.inverse_transform(vector)

class comment_dataset(Dataset):
    '''returns an iterable Dataset that contains the comment vectors computed via a bag of words average of
        onehot encoding of all the words in data'''

    def __init__(self, data):
        self.encoding = corpus_to_one_hot(data)
        self.data = data
        self.nlp = spacy.load('en')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        comment = self.data.loc[index, 'comment']
        label = self.data.loc[index, 'label'].astype('int')
        vector = torch.tensor(create_comment_vector(comment, self.encoding, self.nlp)[0])
        return vector, label#, comment

    def inverse_lookup(self, comment_vector):
        comment_vector = np.asarray(comment_vector)
        non_zero_indicies = np.nonzero(comment_vector)
        non_zero_indicies = non_zero_indicies[0]
        words = []
        for i in range(len(non_zero_indicies)):
            vector = np.zeros((1, self.encoding.dimension()))
            vector[0,non_zero_indicies[i]] = 1
            word = self.encoding.inverse_lookup(vector)
            words.append(word[0][0])
        return words



        return vector, label


class comment_dataset_with_encoding(Dataset):
    '''need this to create test/train datasets from an encoding that has been made from the union of the test and train comment_dataset
        otherwise test/train vectors will have different sizes. workflow follows:
        1. encode word vectors with encoding = corpus_to_one_hot(train comments + test comments)
        2. make train/test datasets with: train = comment_dataset_with_encoding(train, encoding) ; test = //(test, encoding) '''

    def __init__(self, data, encoding):
        self.encoding = encoding
        self.data = data
        self.nlp = spacy.load('en')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        comment = self.data.loc[index, 'comment']
        label = self.data.loc[index, 'label'].astype('int')
        vector = torch.tensor(create_comment_vector(comment, self.encoding, self.nlp)[0])
        return vector, label, comment

    def inverse_lookup(self, comment_vector):
        comment_vector = np.asarray(comment_vector)
        non_zero_indicies = np.nonzero(comment_vector)
        non_zero_indicies = non_zero_indicies[0]
        words = []
        for i in range(len(non_zero_indicies)):
            vector = np.zeros((1, self.encoding.dimension()))
            vector[0,non_zero_indicies[i]] = 1
            word = self.encoding.inverse_lookup(vector)
            words.append(word[0][0])
        return words



        return vector, label

###### fasttext skipgram embedding ##########

class fasttext_word_embedding(Dataset):

    def __init__(self, path_to_fasttext_model, data):
        self.model = fast.load_model(path_to_fasttext_model)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        comment = self.data.iloc[index, 1]
        label = self.data.iloc[index, 0]
        vector = self.model.get_word_vector(comment)
        return vector, label, comment 



##using fasttext with output:
## training_word_embeddings: ./fasttext skipgram -input *corpus cleaned by this module* -output *where you want model parameters saved* (-minn * * -maxn * * -dim * *)
## nearest neighbor queries: ./fastteext nn *location of word embedding*.bin
