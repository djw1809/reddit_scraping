import pandas as pd
import numpy as np
import re
import spacy
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision
import preprocessing_fasttext as pre
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import sklearn
from sklearn.metrics import confusion_matrix
from pathlib import Path

def plot_confusion_matrix(cmat, classes, normalize = False):

    if normalize:
        cmat = cmat.astype('float')/ cmat.sum(axis = 1)[:, np.newaxis]

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
        return h

####################### loss functions ##############################

class CrossEntropyLoss_weight(nn.Module): #Defines cross entropy loss with weight by composing nn.NLLLoss with F.log_softmax in lieu of just using built in CrossEntropyLoss for speed considerations'''

        def __init__(self, weight=None):
            super().__init__()       #inhert all methods of nn.Module (in particular forward())
            self.loss = nn.NLLLoss(weight) #grab the built in log loss

        def forward(self, outputs, targets):  #define the forward method - didnt just use torch's built in cross entropy loss b/c F.log_softmax is faster then log(softmax()) - see torch docs
            return self.loss(F.log_softmax(outputs,dim=1), targets)


#######################training_calls##################################

def train_binary_text_classifier(train_data, test_data, epochs, batch_size, plot, main_folder, folder, filename):

    corpus = train_data.append(test_data)
    corpus.index = range(len(corpus))
    encoding = pre.corpus_to_one_hot(corpus)

    training_dataset = pre.comment_dataset_with_encoding(train_data, encoding)
    test_dataset = pre.comment_dataset_with_encoding(test_data, encoding)

    main_folder_path = Path(main_folder)
    training_run_folder_path = Path(folder)
    final_path = Path(main_folder_path/training_run_folder_path/filename)


    model = linear_model(encoding.dimension(), 2)

    optimizer = optim.Adam(model.parameters(), lr = 1e-2)

    training_loader = DataLoader(training_dataset, shuffle = True, num_workers = 1, batch_size = batch_size)
    test_loader = DataLoader(test_dataset, shuffle = True, num_workers = 1, batch_size = batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loss = CrossEntropyLoss_weight()

    if plot:
        loss_data = np.zeros((epochs)) #empty arrays to store data for plotting in
        accuracy_data = np.zeros((epochs))
        val_accuracy_data = np.zeros((epochs))
        confusion_matrix_ = np.zeros((2,2))
    else:
        pass

    for epoch in range(epochs):

        model.train()

        running_loss = 0
        running_corrects = 0
        running_val_corrects = 0

        for inputs, labels  in training_loader:
            #inputs = inputs.to(device)
            #labels = labels.to(device)
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

        epoch_loss = running_loss / len(training_dataset)
        epoch_corrects = running_corrects / len(training_dataset)

        model.eval()

        for inputs, labels  in test_loader:
            #inputs = inputs.to(device)
            #labels = labels.to(device)
            inputs = inputs.float()
            labels = labels.long()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            running_val_corrects += torch.sum(preds == labels.data).item()
           # confusion_matrix_ += confusion_matrix(labels, preds, labels = None)

        epoch_val_accuracy = running_val_corrects/len(test_dataset)

        if plot:
            loss_data[epoch] = epoch_loss
            accuracy_data[epoch] = epoch_corrects
            val_accuracy_data[epoch] = epoch_val_accuracy

        print(' Loss: {:.4f} Accuracy: {:.4f} Val_Accuracy : {:.4f}'.format(epoch_loss, epoch_corrects, epoch_val_accuracy))

    if plot:
        plt.clf()
        plt.xlabel('epochs')
        plt.plot(range(epochs), loss_data, 'bo')
        plt.plot(range(epochs), accuracy_data, 'ro')
        plt.plot(range(epochs), val_accuracy_data, 'go')

        plt.savefig(main_folder+ '/' +folder+'/'+'plots.png')
        plt.clf()
        #plot_confusion_matrix(confusion_matrix_, ['label1', 'label2'], normalize = False)
        #plt.savefig(filename+'_confusion_matrix.png')

    torch.save(model.state_dict(), main_folder_path/training_run_folder_path/'model')
    np.savetxt(main_folder_path/training_run_folder_path/'loss_data', loss_data, delimiter = ',')
    np.savetxt(main_folder_path/training_run_folder_path/'accuracy_data', accuracy_data, delimiter =',')
    np.savetxt(main_folder_path/training_run_folder_path/'val_accuracy_data', val_accuracy_data, delimiter = ',')
