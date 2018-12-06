import pandas as pd
import models
from pathlib import Path
import json
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np

##model choice
model_choice = 1


##input data
train = pd.read_csv('../Data/test_datasets/small_hillary_donald_train_11092018', sep ='\t')
test = pd.read_csv('../Data/test_datasets/small_hillary_donald_test_11092018', sep = '\t')

for i in train.index:
    if train.loc[i, 'label'] == -1:
        train.loc[i, 'label'] = 0
for i in test.index:
    if test.loc[i, 'label'] == -1:
        test.loc[i, 'label'] = 0

#model parameters
parameter_dict = {}

parameter_dict['epochs'] = 1
parameter_dict['learning_rate'] = 1e-2
parameter_dict['weight'] = None
parameter_dict['batch size'] = 2
parameter_dict['num_workers'] = 1
parameter_dict['model_storage_path'] = '../../saved_models/classification_models'
parameter_dict['results_folder_name'] ='../results'
parameter_dict['filenames'] = 'test'

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
    model, loss_data, accuracy_data, val_accuracy_data = models.train_binary_text_classifier(train, test, parameter_dict['epochs'],  parameter_dict['num_workers'], parameter_dict['batch size'], parameter_dict['learning_rate'], parameter_dict['weight'])

if model_choice == 2:
    parameter_dict['word_embedding_path'] = '../../saved_models/word_embeddings/blah'

    model, loss_data, accuracy_data, val_accuracy_data = models.train_binary_text_classifier_fasttext(train, test, parameter_dict['word_embedding_path'], parameter_dict['epochs'], parameter_dict['num_workers'], parameter_dict['batch size'], parameter_dict['learning_rate'], parameter_dict['weight'])



#save outputs in save locations
epochs = parameter_dict['epochs']
plt.clf()
plt.xlabel('epochs')
plt.plot(range(epochs), loss_data, 'bo')
plt.plot(range(epochs), accuracy_data, 'ro')
plt.plot(range(epochs), val_accuracy_data, 'go')
plt.savefig(parameter_dict['results_folder_name']+ '/' +parameter_dict['filenames'] + '/'+'plots.png')
plt.clf()
#plot_confusion_matrix(confusion_matrix_, ['label1', 'label2'], normalize = False)
#plt.savefig(filename+'_confusion_matrix.png')
torch.save(model.state_dict(), final_model_storage_path/'model')
np.savetxt(main_folder_path/training_run_folder_path/'loss_data', loss_data, delimiter = ',')
np.savetxt(main_folder_path/training_run_folder_path/'accuracy_data', accuracy_data, delimiter =',')
np.savetxt(main_folder_path/training_run_folder_path/'val_accuracy_data', val_accuracy_data, delimiter = ',')

with open(final_results_path/Path('parameters.json'), 'w') as jsonFile:
    json.dump(parameter_dict, jsonFile)
