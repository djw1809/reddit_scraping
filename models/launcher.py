import pandas as pd
import models
from pathlib import Path
import json
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
from models import plot_confusion_matrix
import matplotlib.patches as mpatches

##model choice
model_choice =2

##input data
train = pd.read_csv('../Data/test_datasets/small_hillary_donald_train_shuffled_12122018', sep = '\t')#'../Data/test_datasets/big_hillary_donald_train_not_screened_120718'
test = pd.read_csv('../Data/test_datasets/small_hillary_donald_test_shuffled_12122018', sep = '\t')#'../Data/test_datasets/big_hillary_donald_test_not_screened_120718'

for i in train.index:
    if train.loc[i, 'label'] == -1:
        train.loc[i, 'label'] = 0
for i in test.index:
    if test.loc[i, 'label'] == -1:
        test.loc[i, 'label'] = 0

train = train.dropna(how = 'any') #drop missing values from data if there are any
test = test.dropna(how = 'any')

train.index = range(len(train))
test.index = range(len(test))
#model parameters
parameter_dict = {}

parameter_dict['epochs'] = 10
parameter_dict['learning_rate'] = 1e-4
parameter_dict['weight'] = None
parameter_dict['batch size'] = 2
parameter_dict['num_workers'] = 1
parameter_dict['model_storage_path'] = '../../saved_models/classification_models'
parameter_dict['results_folder_name'] ='../results'
parameter_dict['filenames'] = 'small_hillary_donald_giant_corpus_shuffled_121218'


#save locations
model_storage_path = Path(parameter_dict['model_storage_path'])
main_folder_path = Path(parameter_dict['results_folder_name'])
training_run_folder_path = Path(parameter_dict['filenames'])

final_results_path = Path(main_folder_path/training_run_folder_path)
final_results_path.mkdir(parents = True, exist_ok = True)

final_model_storage_path = Path(model_storage_path/training_run_folder_path)
final_model_storage_path.mkdir(parents = True, exist_ok = True)



#training calls
if model_choice == 1:
    model, loss_data, accuracy_data, val_accuracy_data, confusion_matricies_train, confusion_matricies_test = models.train_binary_text_classifier(train, test, parameter_dict['epochs'],  parameter_dict['num_workers'], parameter_dict['batch size'], parameter_dict['learning_rate'], parameter_dict['weight'])

if model_choice == 2:
    parameter_dict['word_embedding_path'] = '../../saved_models/word_embeddings/giant_corpus_cleanest.bin'

    model, loss_data, accuracy_data, val_accuracy_data, confusion_matrix_train, confusion_matricies_test = models.train_binary_text_classifier_fasttext(train, test, parameter_dict['word_embedding_path'], parameter_dict['epochs'], parameter_dict['num_workers'], parameter_dict['batch size'], parameter_dict['learning_rate'], parameter_dict['weight'])

#save everything EXCEPT model parameters (large file) in a named folder inside a results folder in the git repo - model should be saved somewhere outside of git repo
train.to_csv(main_folder_path/training_run_folder_path/'training_set.csv')
test.to_csv(main_folder_path/training_run_folder_path/'test_set.csv')
torch.save(model.state_dict(), final_model_storage_path/'model')
np.savetxt(main_folder_path/training_run_folder_path/'loss_data', loss_data, delimiter = ',')
np.savetxt(main_folder_path/training_run_folder_path/'accuracy_data', accuracy_data, delimiter =',')
np.savetxt(main_folder_path/training_run_folder_path/'val_accuracy_data', val_accuracy_data, delimiter = ',')

with open(final_results_path/Path('parameters.json'), 'w') as jsonFile:
    json.dump(parameter_dict, jsonFile)

pickle_file_1 = open(main_folder_path/training_run_folder_path/'confusion_matricies_test.pickle', 'wb')
pickle.dump(confusion_matricies_test, pickle_file_1)
pickle_file_1.close()

pickle_file_2 = open(main_folder_path/training_run_folder_path/'confusion_matricies_train.pickle', 'wb')
pickle.dump(confusion_matricies_train, pickle_file_2)
pickle_file_2.close()



epochs = parameter_dict['epochs']
plt.clf()
plt.xlabel('epochs')
plt.plot(range(epochs), loss_data, 'bo')
plt.plot(range(epochs), accuracy_data, 'ro')
plt.plot(range(epochs), val_accuracy_data, 'go')

blue_patch = mpatches.Patch(color = 'blue', label = 'loss')
red_patch = mpatches.Patch(color = 'red', label = 'train accuracy')
green_patch = mpatches.Patch(color = 'green', label = 'test accuracy')

plt.legend(handles = [blue_patch, red_patch, green_patch])

plt.savefig(parameter_dict['results_folder_name']+ '/' +parameter_dict['filenames'] + '/'+'plots.png')
plt.clf()

plot_confusion_matrix(confusion_matricies_test[parameter_dict['epochs']], ['trump', 'hillary'], normalize = False)
plt.savefig(parameter_dict['results_folder_name'] + '/' + parameter_dict['filenames'] + '/' + 'confusion_matrix_test.png')
plt.clf()

plot_confusion_matrix(confusion_matricies_train[parameter_dict['epochs']], ['trump', 'hillary'], normalize = False)
plt.savefig(parameter_dict['results_folder_name'] + '/' + parameter_dict['filenames'] + '/' + 'confusion_matrix_train.png')
plt.clf()
