from sklearn.metrics import f1_score,recall_score,precision_score,confusion_matrix,accuracy_score

import torch
import torch.nn as nn
import copy
import random
import pandas as pd
import numpy as np
from tqdm import trange
import pickle
import json
import sys
import time
import argparse
from sklearn.model_selection import train_test_split

sys.path.append("classes")
sys.path.append("../preprocessing")
from config_parameters import parser
from loss_functions import NuLogsyLossCompute
from model import *
from networks import *
from tokenizer import *
from data_loader import *
from eval_preprocess import EvaluationPreprocessor
from prototype import get_prototypes
from collections import defaultdict
import torch.nn.functional as F
import pickle
import spacy
import os
import glob

from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import mlflow
#from model_training import extract_load, tokenization_dataset

from utils import save_tokenization, tokenize_data, \
                  preprocess_data, top_ranked_repos, extract_load, read_data, \
                  convert_normal_anomaly, convert_error_info, convert_error_warning, \
                  convert_info_warning, convert_error_info_warning, \
                  run_eval, map_prediction_class, convert_tokenizer_to_explainer_data

nlp = spacy.load("en_core_web_sm")

def extract_rules_train(data):
    stats_ = defaultdict(dict)
    log_messages = data.log_messages
    for idx, doc in enumerate(log_messages):
        doc = nlp(doc)
        stri = []
        for token in doc:
            stri.append(token.pos_)
        stats_[idx] = "_".join(sorted(dict.fromkeys(stri)))
        if idx % 1000 == 0:
            print(idx)
    q = pd.DataFrame(stats_, index=[0]).T
    q = q.reset_index()

    unique_rules = np.unique(q.iloc[:, 1])
    return unique_rules


def predict(data, train_rules):
    stats_ = defaultdict(dict)
    log_messages = data.log_messages
    for idx, doc in enumerate(log_messages):
        doc = nlp(doc)
        stri = []
        for token in doc:
            stri.append(token.dep_)
        stats_[idx] = "_".join(sorted(dict.fromkeys(stri)))
        if idx % 1000 == 0:
            print(idx)
    q = pd.DataFrame(stats_, index=[0]).T
    q = q.reset_index()
    predictions = []

    for idx in range(q.shape[0]):
        flag = True
        for rule in train_rules:
            if rule == q.iloc[idx, 1]:
                predictions.append(0)
                flag = False
                break
        if flag == True:
            predictions.append(1)
    return predictions

def syntactic_rule(train_data, test_data, test_labels):
    train_data = train_data[train_data.log_level==0]
    train_rules = extract_rules_train(train_data)
    test_prediction = predict(test_data, train_rules)
    test_data["syntactic_prediction"] = test_prediction
    return test_labels, np.array(test_prediction), test_data

class FCLayer(nn.Module):
    
    def __init__(self, 
                n_dimension, 
                n_targets, 
                max_size, 
                in_features, 
                src_vocab,
                dropout):
        
        super(FCLayer, self).__init__()

        self.layer0 = nn.ModuleList([nn.Linear(in_features, in_features) for i in range(max_size)])
        self.l1 = nn.Linear(n_dimension, n_dimension)
        self.l2 = nn.Linear(n_dimension, n_dimension)
        self.l3 = nn.Linear(n_dimension, n_targets)
        self.max_size = max_size
        self.activation = torch.tanh

    def forward(self, x):
        
        x = x.reshape(-1, 50, 16)

        out = []
        for idx in range(self.max_size):
            out.append(self.layer0[idx](x[:, idx, :]))
        x = torch.cat(out, dim=1)

        x = self.activation(self.l1(x))
        x = self.activation(self.l2(x))
        x = self.l3(x)
        return x


parser = argparse.ArgumentParser(description='Config file for evaluation.')
                                    
         
parser.add_argument('--train_data_path', 
                    type=str,
                    default = '../../3_preprocessed_data/lingustic_properties_data/merged_top10_pos.csv',
                    help='Path where we store data for evaluation')

parser.add_argument('--model_path', 
                    type=str,
                    default = '../../5_results/sytactic_evaluation/lignustic_evaluation_model.pth',
                    help='Path where the model that we want to evaluate is stored')

parser.add_argument('--mlflow_model',
                    #type = bool,
                    #default = False,
                    default = '',
                    help='Indicate if model is loaded from MLFLOW')

parser.add_argument('--device',
                    type=str,
                    default = 'cpu',
                    choices = ['cpu', 'gpu'],
                    help = 'Run script either on gpu or cpu')


args = parser.parse_args()

mlflow.set_experiment("/Erekle-Evaluation-KN")

if __name__ == '__main__':

    device = args.device

    scenario = 'info_error_warning'
    epoch = ''
    seed = 0

    label_col = "log_level"
    tokenizer_path = '../../5_results/sytactic_evaluation/tokenizer.pickle'
    df = pd.read_csv("../../3_preprocessed_data/lingustic_properties_data/merged_top10_pos.csv")
    df = df.loc[:, ["dep", "label"]]
    df.columns = ["log_message", label_col]

    df = df.reset_index()
    df = df.drop(["index"], axis=1)

    print("Currently exploting the classes ", np.unique(df.log_level))
    load, labels = extract_load(df)

    class_count = df.groupby(label_col).count().reset_index()[label_col]

    label_mapper = {class_count.index[i]: i for i in range(len(class_count))}
    tokenized, labels_tokenized, tokenizer = tokenize_data(load, labels, label_mapper, tokenizer_path)
    assert len(tokenized) == df.shape[0], "Some data samples have been lost during tokenization. Take care of this."

    from sklearn.model_selection import StratifiedShuffleSplit

    # seed = 10
    stSampling = StratifiedShuffleSplit(1, test_size=0.3, random_state=seed)

    load_train1 = np.array(tokenized, dtype=object)
    load_train_labels1 = np.array(labels_tokenized)

    df1 = copy.copy(df)
    
    if not args.mlflow_model:    
        model = torch.load(args.model_path, map_location=torch.device(device))
        model.eval()
    else:
        #model_uri = "file:///root/project/log_level_estimation/4_analysis/scripts/mlruns/3/{}/artifacts/pytorch_model".format(args.model_path)
        model_uri = "runs:/{}/pytorch_model".format(args.model_path)
        model = mlflow.pytorch.load_model(model_uri)
        model.eval() 

    if device =="gpu":
        #class_weights=torch.FloatTensor(weights).cuda()
        cross_entropoy_loss = nn.CrossEntropyLoss().cuda()
    else:
        #class_weights = torch.FloatTensor(weights)
        cross_entropoy_loss = nn.CrossEntropyLoss()

    loss_f = cross_entropoy_loss
    polars = None

    in_features=16
    dropout=0.05
    max_len=50
    n_targets = 3
    learning_rate = 1e-4
    src_vocab = tokenizer.n_words
    n_epochs = 10
    batch_size = 256
    pad_len = 16

    #print(model)
    #print(fcl_nn)    

    for train_index, test_index in stSampling.split(load_train1, load_train_labels1):
        
        if device =="gpu":
            #class_weights=torch.FloatTensor(weights).cuda()
            cross_entropoy_loss = nn.CrossEntropyLoss().cuda()
        else:
            #class_weights = torch.FloatTensor(weights)
            cross_entropoy_loss = nn.CrossEntropyLoss()
        
        load_train = load_train1[train_index]
        load_train_labels = load_train_labels1[train_index]
        load_test = load_train1[test_index]
        load_test_labels = load_train_labels1[test_index]

        train_dataloader, test_dataloader = create_data_loaders(load_train, load_train_labels, load_test,  load_test_labels, pad_len, batch_size)

        fcl_nn = FCLayer(in_features*max_len, n_targets, max_len, in_features, src_vocab, dropout)

        f_loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(fcl_nn.parameters(), lr = learning_rate)
        alpha = 0.1
        temperature = 10 

        fcl_nn.train()
        total_loss = 0
        start = time.time()
        
        for epoch in range(1, 1 + n_epochs):
            
            for i, batch in enumerate(train_dataloader):
                load, y = batch
                if polars is not None:
                    y = polars[y.numpy()]
                    y = torch.autograd.Variable(y).cuda()

                # Teacher And Student
                if device == "gpu":
                    teacher_preds = model.forward(load.cuda().long())
                else:
                    teacher_preds = model.forward(load.long())

                #print(teacher_preds.shape)
                #print(student_preds.shape)
                embed_train = convert_tokenizer_to_explainer_data(load, model, max_len)
                embed_train = torch.cat(embed_train)

                print(embed_train.shape)
                if device == 'gpu':
                    out = fcl_nn.forward(embed_train.cuda())
                    loss = f_loss(out, y.cuda().long())
                else:
                    out = fcl_nn.forward(embed_train)
                    loss = f_loss(out, y.long())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss
                elapsed = time.time() - start
                
                try:
                    if i % 5 == 0:
                        print("Epoch %d Train Step: %d / %d Loss: %f" %
                            (epoch, i, len(train_dataloader), loss), end='\r')

                        with torch.no_grad():
                            for test_set in test_dataloader:
                                test_load, y_test = test_set
                                embed_test = convert_tokenizer_to_explainer_data(test_load, model, max_len)
                                embed_test = torch.cat(embed_test)

                                out = fcl_nn.forward(embed_test)
                                tmp = out.detach().cpu().numpy()
                                preds = list(np.argmax(tmp, axis=1))
                                #print(preds, y)
                                print(f"Accuracy:{round(accuracy_score(preds, y_test), 2)}")
                                
                except KeyboardInterrupt:
                    with mlflow.start_run():
                        mlflow.pytorch.log_model(fcl_nn, "pytorch_model")
                    break

            print("Epoch %d Train Step: %d / %d Loss: %f" %
                (epoch, i, len(train_dataloader), loss), end='\r')
    
    # with mlflow.start_run():
    #         mlflow.pytorch.log_model(fcl_nn, "pytorch_model")
            
    #         save_tokenization(tokenizer)
    #         mlflow.log_param("Teacher model path", args.model_path)

    torch.save(fcl_nn,"../../5_results/sytactic_evaluation/shap_evaluation_model.pth")
