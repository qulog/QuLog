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

        print(x)
        print(type(x))
        x = x.reshape(-1, max_len, 16)

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
                    default = '../../3_preprocessed_data/filtered_log_df_reduced.csv',
                    help='Path where we store data for evaluation')

parser.add_argument('--model_path', 
                    type=str,
                    default = '../../5_results/models/learning_scenario_train5/50_info_error_warning_model.pth',
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

    if args.mlflow_model:
        ml_run = mlflow.get_run(args.model_path)
        ml_dict = ml_run.to_dictionary()
        scenario = ml_dict['data']['params']['scenario']
        epoch = ml_dict['data']['params']['epochs']

    data_preprocessor = EvaluationPreprocessor()
    data_path = args.train_data_path
    
    PATH_COUNTS = '../../3_preprocessed_data/stars_repos.csv'
    good_bad_hypo = True
    pad_len = 32
    batch_size = 1024

    number_repos_good = 500
    number_bad_repos = 1
    number_validation_repos = 100

    df = read_data(data_path)
    #data_preprocessor.fit(df)
    df.drop_duplicates(subset = 'log_message', inplace = True)
    repositories = df['repo_link']
    #df = data_preprocessor.transform(df)
    df, indecies_to_preserve = preprocess_data(df, scenario)

    repositories = repositories.loc[indecies_to_preserve]
    repositories = repositories.reset_index()
    star_repos = pd.read_csv(PATH_COUNTS)
    train_good_repos, validation_good_repos, bad_repos, good_bad_repos = top_ranked_repos(repositories, star_repos, number_repos_good, number_bad_repos, number_validation_repos, good_bad_hypo=good_bad_hypo)
    
    df = df.loc[good_bad_repos]
    df = df.reset_index()
    df1 = copy.copy(df)
    df1.columns = ["original_index",'log_message', 'repo_topic', 'repo_link', 'file_name', 'log_level']
    df1 = df1.reset_index().iloc[:, :2]
    df1.index = df1.original_index
    df = df.drop('index', axis=1)

    #testing
    #df.to_csv('../../../my_train.csv')

    load, labels = extract_load(df)
    class_count = df.groupby("log_level").count()['log_message']
    label_mapper = {class_count.index[i]:i for i in range(len(class_count))}

    #tokenized, labels_tokenized, tokenizer = tokenization_dataset(df, load, labels, label_mapper)
    tokenized, labels_tokenized, tokenizer = tokenize_data(load, labels, label_mapper, args.model_path)

    assert len(tokenized) == df.shape[0], "Some data samples have been lost during tokenization. Take care of this."

    load_train = np.array(tokenized, dtype=object)[df1.loc[train_good_repos].iloc[:, 0].values]
    load_train_labels = np.array(labels_tokenized)[df1.loc[train_good_repos].iloc[:, 0].values]

    load_test_good_repos = np.array(tokenized, dtype=object)[df1.loc[validation_good_repos].iloc[:, 0].values]
    load_test_good_repos_labels = np.array(labels_tokenized)[df1.loc[validation_good_repos].iloc[:, 0].values]

    load_test_bad_repos = np.array(tokenized, dtype=object)[df1.loc[bad_repos].iloc[:, 0].values]
    load_test_bad_repos_labels = np.array(labels_tokenized)[df1.loc[bad_repos].iloc[:, 0].values]

    train_dataloader, test_dataloader_good_repos = create_data_loaders(load_train, load_train_labels, load_test_good_repos,  load_test_good_repos_labels, pad_len, batch_size)


    
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

    fcl_nn = FCLayer(in_features*max_len, n_targets, max_len, in_features, src_vocab, dropout)

    #print(model)
    #print(fcl_nn)    

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
            print(type(load))
            if device == "gpu":
                teacher_preds = model.forward(load.cuda().long())
            else:
                teacher_preds = model.forward(load.long())

            embed_train = convert_tokenizer_to_explainer_data(load, model, max_len)
            embed_train = torch.cat(embed_train)

            print(embed_train.shape)
            print(embed_train)
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
                        for test_set in test_dataloader_good_repos:
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
    
    with mlflow.start_run():
            mlflow.pytorch.log_model(fcl_nn, "pytorch_model")
            
            save_tokenization(tokenizer)
            mlflow.log_param("Teacher model path", args.model_path)


                                        