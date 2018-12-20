import pandas as pd
import numpy as np
import re
import spacy
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision
import datasets as pre
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import sklearn
from sklearn.metrics import confusion_matrix
from pathlib import Path
import itertools

def plot_confusion_matrix(cmat, classes, normalize = False):

    if normalize:
        cmat = cmat.astype('float')/ cmat.sum(axis = 1)[:, np.newaxis]
    cmat = cmat.astype('float')
    plt.imshow(cmat, interpolation = 'nearest', cmap = plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f'
    thresh = cmat.max()/2.

    for i, j in itertools.product(range(cmat.shape[0]), range(cmat.shape[1])):
        plt.text(j, i, format(cmat[i, j], fmt), horizontalalignment = "center", color = "white" if cmat[i,j]> thresh else "black")
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.title('Predicted label')

#######################  models #################################
class linear_model(nn.Module):

    def __init__(self, size1, size2):
        super(linear_model, self).__init__()
        self.fc = nn.Linear(size1, size2)

    def forward(self, x):
        h = self.fc(x)
        return

class one_layer_relu(nn.Module):
    def __init__(self, size_11, size_12, size_21, size_22):
        super(one_layer_relu, self).__init__()
        self.fc1 = nn.Linear(size_11, size_12)
        self.fc2 = nn.Linear(size_12, size_22)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x     


####################### loss functions ##############################

class CrossEntropyLoss_weight(nn.Module): #Defines cross entropy loss with weight by composing nn.NLLLoss with F.log_softmax in lieu of just using built in CrossEntropyLoss for speed considerations'''

        def __init__(self, weight=None):
            super().__init__()       #inhert all methods of nn.Module (in particular forward())
            self.loss = nn.NLLLoss(weight) #grab the built in log loss

        def forward(self, outputs, targets):  #define the forward method - didnt just use torch's built in cross entropy loss b/c F.log_softmax is faster then log(softmax()) - see torch docs
            return self.loss(F.log_softmax(outputs,dim=1), targets)


#######################training_calls##################################

def train_binary_text_classifier(train_data, test_data, epochs, num_workers, batch_size, learning_rate, weight = None):

    corpus = train_data.append(test_data)
    corpus.index = range(len(corpus))
    encoding = pre.corpus_to_one_hot(corpus)

    training_dataset = pre.comment_dataset_with_encoding(train_data, encoding)
    test_dataset = pre.comment_dataset_with_encoding(test_data, encoding)




    model = linear_model(encoding.dimension(), 2)

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    training_loader = DataLoader(training_dataset, shuffle = True, num_workers = num_workers, batch_size = batch_size)
    test_loader = DataLoader(test_dataset, shuffle = True, num_workers = num_workers, batch_size = batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        model.cuda()
        if weight != None:
            weight = weight.to(device)

    loss = CrossEntropyLoss_weight(weight)

    loss_data = np.zeros((epochs)) #empty arrays to store data for plotting in
    accuracy_data = np.zeros((epochs))
    val_accuracy_data = np.zeros((epochs))

    confusion_matricies_test = {}
    confusion_matricies_train = {}


    for epoch in range(epochs):

        model.train()

        confusion_matrix_train_epoch = np.zeros((2,2))
        confusion_matrix_test_epoch = np.zeros((2,2))

        running_loss = 0
        running_corrects = 0
        running_val_corrects = 0

        for inputs, labels  in training_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.float()
            labels = labels.long()
            optimizer.zero_grad()

            #forward
            outputs = model(inputs)
            loss_value = loss(outputs, labels)
            _, preds = torch.max(outputs.data, 1)

            #backwards
            loss_value.backward()
            optimizer.step()

            running_loss += loss_value.item()
            running_corrects += torch.sum(preds == labels.data).item()
            confusion_matrix_train_epoch += confusion_matrix(labels, preds, labels =range(2))

        epoch_loss = running_loss / len(training_dataset)
        epoch_corrects = running_corrects / len(training_dataset)

        model.eval()

        for inputs, labels  in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.float()
            labels = labels.long()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            running_val_corrects += torch.sum(preds == labels.data).item()
            confusion_matrix_test_epoch += confusion_matrix(labels, preds, labels =range(2))

        epoch_val_accuracy = running_val_corrects/len(test_dataset)


        loss_data[epoch] = epoch_loss
        accuracy_data[epoch] = epoch_corrects
        val_accuracy_data[epoch] = epoch_val_accuracy
        confusion_matricies_test[epoch] = confusion_matrix_test_epoch
        confusion_matricies_train[epoch] = confusion_matrix_train_epoch

        print(' Loss: {:.4f} Accuracy: {:.4f} Val_Accuracy : {:.4f}'.format(epoch_loss, epoch_corrects, epoch_val_accuracy))

    return model, loss_data, accuracy_data, val_accuracy_data, confusion_matricies_train, confusion_matricies_test




def train_binary_text_classifier_fasttext(train_data, test_data, model_path, epochs, num_workers, batch_size, learning_rate, weight = None):

    training_dataset = pre.fasttext_word_embedding(model_path, train_data)
    test_dataset = pre.fasttext_word_embedding(model_path, test_data)

    model = linear_model(training_dataset.model.get_dimension(), 2)

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    training_loader = DataLoader(training_dataset, shuffle = True, num_workers = num_workers, batch_size = batch_size)
    test_loader = DataLoader(test_dataset, shuffle = True, num_workers = num_workers, batch_size = batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    if torch.cuda.is_available():
        model.cuda()
        if weight != None:
            weight = weight.to(device)


    loss = CrossEntropyLoss_weight(weight)

    loss_data = np.zeros((epochs)) #empty arrays to store data for plotting in
    accuracy_data = np.zeros((epochs))
    val_accuracy_data = np.zeros((epochs))

    confusion_matricies_test = {}
    confusion_matricies_train = {}


    for epoch in range(epochs):
        confusion_matrix_test_epoch = np.zeros((2,2))
        confusion_matrix_train_epoch = np.zeros((2,2))
        model.train()

        running_loss = 0
        running_corrects = 0
        running_val_corrects = 0

        for inputs, labels  in training_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.float()
            labels = labels.long()
            optimizer.zero_grad()

            #forward
            outputs = model(inputs)
            loss_value = loss(outputs, labels)
            _, preds = torch.max(outputs.data, 1)

            #backwards
            loss_value.backward()
            optimizer.step()

            running_loss += loss_value.item()
            running_corrects += torch.sum(preds == labels.data).item()

            confusion_matrix_train_epoch += confusion_matrix(labels, preds, labels = range(2))

        epoch_loss = running_loss / len(training_dataset)
        epoch_corrects = running_corrects / len(training_dataset)

        model.eval()

        for inputs, labels  in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.float()
            labels = labels.long()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            running_val_corrects += torch.sum(preds == labels.data).item()
            confusion_matrix_test_epoch += confusion_matrix(labels, preds, labels = range(2))

        epoch_val_accuracy = running_val_corrects/len(test_dataset)

        confusion_matricies_test[epoch] = confusion_matrix_test_epoch
        confusion_matricies_train[epoch] = confusion_matrix_train_epoch
        loss_data[epoch] = epoch_loss
        accuracy_data[epoch] = epoch_corrects
        val_accuracy_data[epoch] = epoch_val_accuracy

        print(' Loss: {:.4f} Accuracy: {:.4f} Val_Accuracy : {:.4f}'.format(epoch_loss, epoch_corrects, epoch_val_accuracy))

    return model, loss_data, accuracy_data, val_accuracy_data, confusion_matricies_train, confusion_matricies_test
