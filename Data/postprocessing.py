import pandas as pd
import numpy as np
import re
import spacy
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader
import torch
import fastText.FastText as fast


class FastTextNN:

    def __init__(self, ft_model, ft_matrix=None):
        self.ft_model = ft_model
        self.ft_words = ft_model.get_words()
        self.word_frequencies = dict(zip(*ft_model.get_words(include_freq=True)))

        self.ft_matrix = np.empty((len(self.ft_words), ft_model.get_dimension()))
        for i, word in enumerate(self.ft_words):
            self.ft_matrix[i,:] = ft_model.get_word_vector(word)

    def find_nearest_neighbor(self, query, vectors, n=10,  cossims=None):
        """
        query is a 1d numpy array corresponding to the vector to which you want to
        find the closest vector
        vectors is a 2d numpy array corresponding to the vectors you want to consider

        cossims is a 1d numpy array of size len(vectors), which can be passed for efficiency
        returns the index of the closest n matches to query within vectors and the cosine similarity (cosine the angle between the vectors)

        """
        if cossims is None:
            cossims = np.matmul(vectors, query, out=cossims)

        norms = np.sqrt((query**2).sum() * (vectors**2).sum(axis=1))
        cossims = cossims/norms
        result_i = np.argpartition(-cossims, range(n+1))[1:n+1]     ##flip order of the sort of cossims have the indices of the n+1 largest entries be in order and first in the output, all smaller indices are behind these in output and are out of order 
        return list(zip(result_i, cossims[result_i]))

    def nearest_words(self, vector = None, word = None, n=10, word_freq=None):

        if word != None:
            result = self.find_nearest_neighbor(self.ft_model.get_word_vector(word), self.ft_matrix, n=n)
        if vector != None:
            result = self.find_nearest_neighbor(vector, self.ft_matrix, n=n)

        if word_freq:
            return [(self.ft_words[r[0]], r[1]) for r in result if self.word_frequencies[self.ft_words[r[0]]] >= word_freq]
        else:
            return [(self.ft_words[r[0]], r[1]) for r in result]