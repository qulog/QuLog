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
from utils import tokenize_data

from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import mlflow
#from model_training import extract_load, tokenization_dataset

def preprocess_data(df, scenario, verbose=True):

    if verbose:
        print("Filtering the special characters in the dataframe!")
    df['log_message'] = df['log_message'].str.replace("\<\*\>", " ")
    df['log_message'] = df['log_message'].str.replace("\[STR\]", " ")
    df['log_message'] = df['log_message'].str.replace("\[NUM\]", " ")

    if verbose:
        scenario_list = scenario.upper().split('_')
        print(f"Converting the classes into required categories. Pair of ({' '.join(scenario_list)}). ")

    if scenario=="error_warning":
        df.loc[:, 'log_level'] = df.loc[:, 'log_level'].apply(lambda x: convert_error_warning(x))
    elif scenario == "info_warning":
        df.loc[:, 'log_level'] = df.loc[:, 'log_level'].apply(lambda x: convert_info_warning(x))
    elif scenario == "info_error":
        df.loc[:, 'log_level'] = df.loc[:, 'log_level'].apply(lambda x: convert_error_info(x))
    elif scenario=="info_error_warning":
        df.loc[:, 'log_level'] = df.loc[:, 'log_level'].apply(lambda x: convert_error_info_warning(x))
    else:
        print("Insert a valid scenario, one in error_warning, info_warning, info_error, info_error_warning")
        exit(-1)

    if verbose:
        print("Prior removing (DEBUG, LOG and TRACE) ", df.shape)
    df = df[df['log_level'] != 'debug']
    df = df[df['log_level'] != 'log']
    df = df[df['log_level'] != 'trace']

    if verbose:
        print("Size after removal ", df.shape)
    indecies_to_preserve = df.index
    df = df.reset_index()
    df = df.drop("index", axis=1)
    return df, indecies_to_preserve

def extract_load(df):
    print("Split descriptive and target data into numpay arrays.")
    load = df['log_message'].values
    labels = df['log_level'].values
    return load, labels

def tokenization_dataset(df, load, labels, label_mapper):
    tokenizer = LogTokenizer()
    tokenized = []
    for i in trange(0, len(df)):
        tokenized.append(np.array(tokenizer.tokenize(df['log_message'][i])))
    labels_tokenized = [label_mapper[label] for label in labels]
    return tokenized, labels_tokenized, tokenizer

def read_data(path):
    print("Reading data at path ", path)
    try:
        return pd.read_csv(path).drop(columns=["Unnamed: 0"])
    except:
        return pd.read_csv(path)

def convert_normal_anomaly(x):
    if x == 'trace':
        return 'trace'
    elif x == 'warn':
        return 'warning'
    elif x == 'warning':
        return 'warning'
    elif x == 'info':
        return 'normal'
    elif x == 'debug':
        return 'debug'
    elif x == 'log':
        return 'log'
    else:
        return 'anomaly'

def convert_error_info(x):
    if x == 'trace':
        return 'trace'
    elif x == 'warn':
        return 'log'
    elif x == 'warning':
        return 'log'
    elif x == 'info':
        return 'info'
    elif x == 'debug':
        return 'debug'
    elif x == 'log':
        return 'log'
    else:
        return 'error'

def convert_error_warning(x):
    if x == 'trace':
        return 'trace'
    elif x == 'warn':
        return 'warning'
    elif x == 'warning':
        return 'warning'
    elif x == 'info':
        return 'debug'
    elif x == 'debug':
        return 'debug'
    elif x == 'log':
        return 'log'
    else:
        return 'error'


def convert_info_warning(x):
    if x == 'trace':
        return 'trace'
    elif x == 'warn':
        return 'warning'
    elif x == 'warning':
        return 'warning'
    elif x == 'info':
        return 'info'
    elif x == 'debug':
        return 'debug'
    elif x == 'log':
        return 'log'
    else:
        return 'log'

def convert_error_info_warning(x):
    if x == 'trace':
        return 'trace'
    elif x == 'warn':
        return 'warning'
    elif x == 'warning':
        return 'warning'
    elif x == 'info':
        return 'info'
    elif x == 'debug':
        return 'debug'
    elif x == 'log':
        return 'log'
    else:
        return 'error'

def run_eval(dataloader, model, f_loss, polars=None, device="cpu"):
    
    model.eval()
    preds = []

    with torch.no_grad():

        for i, batch in enumerate(dataloader):

            load, y = batch

            if device=="gpu":
                out = model.forward(load.cuda().long())
            else:
                out = model.forward(load.long())

            if isinstance(f_loss, nn.CosineSimilarity):
                x = F.normalize(out, p=2, dim=1)
                x = torch.mm(x, polars.t().cuda())
                pred = x.max(1, keepdim=True)[1].reshape(1, -1)[0]
                preds += list(pred.detach().cpu().numpy())
            else:
                tmp = out.detach().cpu().numpy()
                preds += list(np.argmax(tmp, axis=1))

    return preds

def map_prediction_class(x, reverse_dict):

    try: 
        return reverse_dict[x]
    except:
        return 0


parser = argparse.ArgumentParser(description='Config file for evaluation.')
                                    
         
parser.add_argument('--evaluate_data_path', 
                    type=str,
                    default = '../../3_preprocessed_data/filtered_log_df_reduced.csv',
                    help='Path where we store data for evaluation')

parser.add_argument('--model_path', 
                    type=str,
                    default = '../../5_results/models/mlc_model/MLC_model_DEBUG_INFO_ERROR_WARNING.pth',
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

mlflow.set_experiment("/Erekle-Evaluation5")

if __name__ == '__main__':

    device = args.device

    scenario = 'info_error'
    epoch = ''

    try:    
        model = torch.load(args.model_path, map_location=torch.device(device))
        model.eval()
    except:
        print("Not found in dir, trying MLFLOW model")
        #model_uri = "file:///root/project/log_level_estimation/4_analysis/scripts/mlruns/3/{}/artifacts/pytorch_model".format(args.model_path)
        model_uri = "runs:/{}/pytorch_model".format(args.model_path)
        model = mlflow.pytorch.load_model(model_uri)
        model.eval()

        ml_run = mlflow.get_run(args.model_path)
        ml_dict = ml_run.to_dictionary()
        scenario = ml_dict['data']['params']['scenario']
        epoch = ml_dict['data']['params']['epochs']

    data_path = args.evaluate_data_path
    if data_path[-3:] == 'csv':
        data_path_list = [data_path]
    else:
        data_path_list = glob.glob(data_path + '*.csv')

    for evaluate_data_path in data_path_list:

        data_preprocessor = EvaluationPreprocessor()
        df = read_data(evaluate_data_path)

        data_preprocessor.fit(df)

        df = data_preprocessor.transform(df)

        repositories = df['repo_link']
        df, indecies_to_preserve = preprocess_data(df, scenario)

        df = df.reset_index()
        df1 = copy.copy(df)
        df1.columns = ["original_index",'log_message', 'repo_topic', 'repo_link', 'file_name', 'log_level']
        df1 = df1.reset_index().iloc[:, :2]
        df1.index = df1.original_index
        df = df.drop('index', axis=1)

        load, labels = extract_load(df)
        class_count = df.groupby("log_level").count()['log_message']
        label_mapper = {class_count.index[i]:i for i in range(len(class_count))}
        reverse_label_mapper = {val: key for key, val in label_mapper.items()}

        #label_mapper = {'error': 0, 'info': 1, 'warning': 2}
        #print(df['log_level'].value_counts())
        print(label_mapper)

        #tokenized, labels_tokenized, tokenizer = tokenization_dataset(df, load, labels, label_mapper)
        tokenized, labels_tokenized, tokenizer = tokenize_data(load, labels, label_mapper, args.model_path)

        #assert len(tokenized) == df.shape[0], "Some data samples have been lost during tokenization. Take care of this."
        
        load_eval = np.array(tokenized, dtype=object)[df1.iloc[:, 0].values]
        load_eval_labels = np.array(labels_tokenized)[df1.iloc[:, 0].values]
        #print(len(load_eval), len(load_eval_labels))
        #print(load_eval,load_eval_labels)

        pad_len = 32
        batch_size = 64

        eval_dataloader = create_test_data_loaders(load_eval, load_eval_labels, pad_len, batch_size)

        if device =="gpu":
            #class_weights=torch.FloatTensor(weights).cuda()
            cross_entropoy_loss = nn.CrossEntropyLoss().cuda()
        else:
            #class_weights = torch.FloatTensor(weights)
            cross_entropoy_loss = nn.CrossEntropyLoss()

        loss_f = cross_entropoy_loss
        polars = None

        preds = run_eval(eval_dataloader, model, loss_f, polars, device)

        print(f"Accuracy:{round(accuracy_score(preds, load_eval_labels), 2)}")
        print(f"f1_score:{round(f1_score(preds, load_eval_labels, average='micro'), 2)}")
        print(f"recall_score:{round(recall_score(preds, load_eval_labels, average='micro'), 2)}")
        print(f"precision_score:{round(precision_score(preds, load_eval_labels, average='micro'), 2)}")

        with mlflow.start_run():
            
            mlflow.log_param("model", args.model_path.split('/')[-1])
            mlflow.log_param("mlflow_model", '{}_{}'.format(epoch, scenario))
            mlflow.log_param("data", evaluate_data_path.split('/')[-1])
            mlflow.log_metric("Acc", round(accuracy_score(preds, load_eval_labels), 2))
            mlflow.log_metric("F1", round(f1_score(preds, load_eval_labels, average='micro'), 2))
            mlflow.log_metric("Recall", round(recall_score(preds, load_eval_labels, average='micro'), 2))
            mlflow.log_metric("Precision", round(precision_score(preds, load_eval_labels, average='micro'), 2))

        pred_df = pd.DataFrame()
        pred_df['pred_log_level'] = preds
        pred_df['pred_log_level'] = pred_df['pred_log_level'].apply(lambda x : map_prediction_class(x, reverse_label_mapper)) 
        
        #print(pred_df['pred_log_level'].value_counts())

        real_eval, pred_eval = df['log_level'].value_counts(), pred_df['pred_log_level'].value_counts() 
        for key,val in real_eval.items():
            print(f"Log level {key} Pred/Real {pred_eval[key]}/{val}")


        #print(preds, load_eval_labels)
        #for i in range(len(preds)):
        #    print(df.log_message[i] ,  "->>> Real : ", reverse_label_mapper[load_eval_labels[i]], "Pred : ", reverse_label_mapper[preds[i]])

    