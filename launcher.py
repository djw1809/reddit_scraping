import pandas as pd
import models as model
from pathlib import Path

fasttext = 1 

train = pd.read_csv('small_hillary_donald_train_11092018', sep ='\t')
test = pd.read_csv('small_hillary_donald_test_11092018', sep = '\t')

for i in range(len(train)):
    if train.loc[i, 'label'] == -1:
        train.loc[i, 'label'] = 0
    else:
        pass



for i in range(len(test)):
    if train.loc[i, 'label'] == -1:
        test.loc[i, 'label'] = 0
    else:
        pass


main_folder = Path('results')
training_folder = Path('test_fasttext')

final_path = Path(main_folder/training_folder)
final_path.mkdir(parents = True, exist_ok = True)

if fasttext == 0:
    model.train_binary_text_classifier(train, test, 20, 2, True,'results', 'test1' ,'test1')

if fasttext == 1:
    model.train_binary_text_classifier_fasttext(train, test, 'fastText_models/small.bin', 5, 2, True, 'results', 'test_fasttext', 'test_fasttext')
