# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 16:28:50 2023

@author: Sunny
"""

# Libraries
import pandas as pd
import re
import spacy
from nltk.stem import PorterStemmer
import torch
import torch.optim as optim
from torchtext.legacy import data
from torchtext.legacy.data import Field, LabelField, TabularDataset
import time

   

# -----------------------------------------------------------------------------
# Set up random seed

SEED = 1

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# -----------------------------------------------------------------------------
# Set up field & loading datasets
TEXT = Field(sequential = True, tokenize = "spacy", lower = True)
LABEL = LabelField(dtype = torch.float)
 
train_datafield = [("title", TEXT), 
                   ("abstract", None),
                   ("InformationTheory", LABEL), 
                   ("ComputationalLinguistics", LABEL),
                   ("ComputerVision", LABEL)
                   ]

train_data, test_data = TabularDataset.splits(
    path = "./",
    train = "train.csv", test = "test.csv", format = "csv",
    skip_header = True, fields = train_datafield)

from torchtext.legacy.data import Dataset

def split_dataset(dataset, split_index):
    fields = dataset.fields
    examples = dataset.examples
    top_examples = examples[:split_index]
    remaining_examples = examples[split_index:]
    
    top_dataset = Dataset(top_examples, fields)
    remaining_dataset = Dataset(remaining_examples, fields)
    
    return top_dataset, remaining_dataset

train_data_1000, remaining_train_data = split_dataset(train_data, 1000)


# -----------------------------------------------------------------------------

# Building vocab
MAX_VOCAB_SIZE = 5400

TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
LABEL.build_vocab(train_data)
# -----------------------------------------------------------------------------
# Create iterator

BATCH_SIZE = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
def preprocess_target_label(target_field):
    for example in train_data:
        setattr(example, f"label_{target_field}", getattr(example, target_field))
        
    for example in test_data:
        setattr(example, f"label_{target_field}", getattr(example, target_field))

def generate_label_iterator(dataset, target, validation = True):
    preprocess_target_label(target)
    
    label_attr = f"label_{target}"
    print(label_attr)
    if validation:
        iterators = data.BucketIterator.splits(
            (train_data_1000, valid_data, test_data),
            batch_size = BATCH_SIZE,
            device = device,
            sort_key = lambda x: len(getattr(x, label_attr)),
            sort_within_batch = False)
        return iterators[0], iterators[1], iterators[2]
    else:
        iterators = data.BucketIterator.splits(
            (train_data_1000, test_data),
            batch_size = BATCH_SIZE,
            device = device,
            sort_key = lambda x: len(getattr(x, label_attr)),

            sort_within_batch = False)
        return iterators[0], iterators[1]
    


# train_iterator_IT, test_iterator_IT = generate_label_iterator(train_data, "InformationTheory", validation = False)
# train_iterator_CL, test_iterator_CL = generate_label_iterator(train_data, "ComputationalLinguistics", validation = False)
# train_iterator_CV, test_iterator_CV = generate_label_iterator(train_data, "ComputerVision", validation = False)

train_iterator_IT, validation_IT, test_iterator_IT = generate_label_iterator(train_data_1000, "InformationTheory", validation = True)
train_iterator_CL, validation_CL, test_iterator_CL = generate_label_iterator(train_data_1000, "ComputationalLinguistics", validation = True)
train_iterator_CV, validation_CV, test_iterator_CV = generate_label_iterator(train_data_1000, "ComputerVision", validation = True)




# -----------------------------------------------------------------------------

# Define RNN

import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):

        embedded = self.embedding(text)
        
        output, hidden = self.rnn(embedded)
        
        assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        
        return self.fc(hidden.squeeze(0))

# -----------------------------------------------------------------------------

# Model initialization & optimizer
def generate_model_and_optimizer(embedding_dim=100, hidden_dim=256, output_dim=1, lr=1e-3):
    INPUT_DIM = len(TEXT.vocab)

    model = RNN(INPUT_DIM, embedding_dim, hidden_dim, output_dim)

    optimizer = optim.SGD(model.parameters(), lr=lr)

    model = model.to(device)
    
    return model, optimizer

model_IT, optimizer_IT = generate_model_and_optimizer()
model_CL, optimizer_CL = generate_model_and_optimizer()
model_CV, optimizer_CV = generate_model_and_optimizer()

criterion = nn.BCEWithLogitsLoss()
criterion = criterion.to(device)

# -----------------------------------------------------------------------------

# Evaluation functions

def binary_accuracy(preds, y):

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

# def train(model, iterator, optimizer, criterion):
#     epoch_loss = 0
#     epoch_acc = 0
#     model.train()
#     for batch in iterator:
#         optimizer.zero_grad()
#         predictions = model(batch.label).squeeze(1)
#         loss = criterion(predictions, batch.label)
#         acc = binary_accuracy(predictions, batch.label)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#         epoch_acc += acc.item()
#     return epoch_loss / len(iterator), epoch_acc / len(iterator)

def train(model, iterator, optimizer, criterion, label_field):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
                
        predictions = model(batch.title).squeeze(1)
        
        # Use the specific label field
        loss = criterion(predictions, getattr(batch, label_field))
        
        acc = binary_accuracy(predictions, getattr(batch, label_field))
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, label_field):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.title).squeeze(1)
            
            # Use the specific label field
            loss = criterion(predictions, getattr(batch, label_field))
            
            acc = binary_accuracy(predictions, getattr(batch, label_field))

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
    
import time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# -----------------------------------------------------------------------------

# Training Loop

N_EPOCHS = 5
label_names = ["InformationTheory", "ComputationalLinguistics", "ComputerVision"]
models = [model_IT, model_CL, model_CV]
optimizers = [optimizer_IT, optimizer_CL, optimizer_CV]
iterators = [(train_iterator_IT, test_iterator_IT), (train_iterator_CL, test_iterator_CL), (train_iterator_CV, test_iterator_CV)]


for idx, (label_name, model, optimizer, (train_iterator, test_iterator)) in enumerate(zip(label_names, models, optimizers, iterators)):
    print(f"Training model for {label_name}...")
    best_valid_loss_IT = float("inf")
    best_valid_loss_CL = float("inf")
    best_valid_loss_CV = float("inf")
    
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        
        if label_name == label_names[0]:
            train_loss_IT, train_acc_IT = train(model_IT, train_iterator_IT, optimizers[idx], criterion, label_name)
            test_loss_IT, test_acc_IT = evaluate(model_IT, test_iterator_IT, criterion, label_name)
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            if test_loss_IT < best_valid_loss_IT:
                best_valid_loss_IT = test_loss_IT
                torch.save(models[0].state_dict(), "RNN_model_IT.pt")
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'InformationTheory Test Loss: {test_loss_IT:.3f} | Test Acc: {test_acc_IT*100:.2f}%')
            model_IT.eval()
            y_predict = []
            y_test = []
            with torch.no_grad():
                for batch in test_iterator_IT:
                    predictions = model_IT(batch.title).squeeze(1)
                    rounded_preds = torch.round(torch.sigmoid(predictions))
                    y_predict += rounded_preds.tolist()
                    y_test += batch.InformationTheory.tolist()
                       
        elif label_name == label_names[1]:
            train_loss_CL, train_acc_CL = train(model_CL, train_iterator_CL, optimizers[idx], criterion, label_name)
            test_loss_CL, test_acc_CL = evaluate(model_CL, test_iterator_CL, criterion, label_name )
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            if test_loss_CL < best_valid_loss_CL:
                best_valid_loss_CL = test_loss_CL
                torch.save(models[1].state_dict(), "RNN_model_CL.pt")
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'ComputationalLinguistics Test Loss: {test_loss_CL:.3f} | Test Acc: {test_acc_CL*100:.2f}%')

           
        elif label_name == label_names[2]:
            train_loss_CV, train_acc_CV = train(model_CV, train_iterator_CV, optimizers[2], criterion, label_name)
            test_loss_CV, test_acc_CV = evaluate(model_CV, test_iterator_CV, criterion, label_name)
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            if test_loss_CV < best_valid_loss_CV:
                best_valid_loss_CV = test_loss_CV
                torch.save(models[2].state_dict(), "RNN_model_CV.pt")
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'ComputerVision Test Loss: {test_loss_CV:.3f} | Test Acc: {test_acc_CV*100:.2f}%')

    



from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, matthews_corrcoef
import numpy as np

def evaluate_model(model, iterator, label_field):
    y_predict = []
    y_test = []
    
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.title).squeeze(1)
            rounded_preds = torch.round(torch.sigmoid(predictions))
            y_predict += rounded_preds.tolist()
            y_test += getattr(batch, label_field).tolist()

    y_predict = np.asarray(y_predict)
    y_test = np.asarray(y_test)

    # Compute metrics
    recall = recall_score(y_test, y_predict, average='macro')
    precision = precision_score(y_test, y_predict, average='macro')
    f1score = f1_score(y_test, y_predict, average='macro')
    accuracy = accuracy_score(y_test, y_predict)
    matthews = matthews_corrcoef(y_test, y_predict)

    # Print metrics
    print(f"{label_field}:")
    print(confusion_matrix(y_test, y_predict))
    print('Accuracy:', accuracy)
    print('Macro Precision:', precision)
    print('Macro Recall:', recall)
    print('Macro F1 score:', f1score)
    print('MCC:', matthews)
    print("\n")

# Evaluate models
evaluate_model(model_IT, test_iterator_IT, "InformationTheory")
evaluate_model(model_CL, test_iterator_CL, "ComputationalLinguistics")
evaluate_model(model_CV, test_iterator_CV, "ComputerVision")


"""
Steps:
    1. pre-process
    
    2. embedding
    params:
        - dimensions of hidden layer
        - sentence length
    3. RNNetwork
    params:
        - Number of epoch
        
    4. back-propogation
    params:
        - loss function
        - optimization method
        
    
"""




