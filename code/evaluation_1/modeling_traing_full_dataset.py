from sklearn.metrics import f1_score,recall_score,precision_score,confusion_matrix,accuracy_score

from pylab import *
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
import shap
from sklearn.model_selection import train_test_split

sys.path.append("classes")
sys.path.append("/home/matilda/PycharmProjects/log_level_estimation/TorchLRP")
from loss_functions import NuLogsyLossCompute
from model import *
from networks import *
from tokenizer import *
from data_loader import *
from prototype import get_prototypes
from collections import defaultdict
import torch.nn.functional as F
import pickle
import spacy

from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

class Baseline(nn.Module):
    def __init__(self, n_dimension, n_targets, max_size, d_model):
        super(Baseline, self).__init__()
        self.layer0 = nn.ModuleList([nn.Linear(d_model, d_model) for i in range(max_size)])
        self.l1 = nn.Linear(n_dimension, n_dimension)
        self.l2 = nn.Linear(n_dimension, n_dimension)
        self.l3 = nn.Linear(n_dimension, n_targets)
        self.max_size = max_size
        self.activation = torch.tanh

    def forward(self, input):
        input = input.reshape(-1, 50, 16)
        out = []
        for idx in range(self.max_size):
            out.append(self.layer0[idx](input[:, idx, :]))
        input = torch.cat(out, dim=1)
        input = self.activation(self.l1(input))
        input = self.activation(self.l2(input))
        input = self.l3(input)
        return input


def run_train_baseline(dataloader, model, optimizer, f_loss, epoch, device="cpu"):
    model.train()
    total_loss = 0
    start = time.time()
    for i, batch in enumerate(dataloader):
        load, y = batch
        # print("device")
        if device == "cuda":
            out = model.forward(load.cuda())
        else:
            out = model.forward(load)
        if device == "cuda":

            loss = f_loss(out, y.cuda().long())
        else:
            loss = f_loss(out, y.long())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss
        elapsed = time.time() - start
        if i % 5 == 0:
            print("Epoch %d Train Step: %d / %d Loss: %f" % (epoch, i, len(dataloader), loss), end='\r')
    print("Epoch %d Train Step: %d / %d Loss: %f" % (epoch, i, len(dataloader), loss), end='\r')
    return total_loss / len(dataloader)


def run_test_baseline(dataloader, model, optimizer, f_loss, epoch,  device="cpu"):
    model.eval()
    preds = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            load, y = batch
            if device=="cuda":
                out = model.forward(load.cuda())
            else:
                out = model.forward(load)
            if device=="cuda":
                tmp = out.detach().cpu().numpy()
            else:
                tmp = out.detach().cpu().numpy()
            preds += list(np.argmax(tmp, axis=1))
    return preds

def run_optimizer_baseline(model, train_dataloader, test_dataloader_good_repos, test_dataloader_bad_repos,  load_test_good_repos_labels, load_test_bad_repos_labels, optimizer, n_epochs,cross_entropoy_loss,class_weights, device):
    conf_matrix_good = []
    conf_matrix_bad = []
    preds = []
    best_f1_score = 0
    best_conf_matrix = []
    best_model = []
    best_preds = []

    for epoch in range(1, 1 + n_epochs):
        loss = run_train_baseline(train_dataloader, model, optimizer, cross_entropoy_loss, epoch, device=device)

        print("Epoch %d Train Loss: %f" % (epoch, loss), " " * 30)
        start_time = time.time()

        print("----------GOOD REPOS----------")
        preds1 = run_test_baseline(test_dataloader_good_repos, model, optimizer, cross_entropoy_loss, epoch, device=device)
        print(f"Accuracy:{round(accuracy_score(preds1, load_test_good_repos_labels), 2)}")
        print(f"f1_score:{round(f1_score(preds1, load_test_good_repos_labels, average='binary'), 2)}")
        print(f"recall_score:{round(recall_score(preds1, load_test_good_repos_labels, average='binary'), 2)}")
        print(f"precision_score:{round(precision_score(preds1, load_test_good_repos_labels, average='binary'), 2)}")
        print(f"confusion matrix: ", confusion_matrix(preds1, load_test_good_repos_labels))
        conf_matrix_good.append(confusion_matrix(preds1, load_test_good_repos_labels))
        calc_f1_score = f1_score(preds1, load_test_good_repos_labels, average='binary')
        if best_f1_score < calc_f1_score:
            best_f1_score = calc_f1_score
            best_conf_matrix = confusion_matrix(preds1, load_test_good_repos_labels)
            best_model = model
            best_preds = preds1

        # print("----------BAD REPOS----------")
        #
        # preds = run_test_baseline(test_dataloader_bad_repos, model, optimizer, cross_entropoy_loss, epoch, device=device)
        # print(f"Accuracy:{round(accuracy_score(preds, load_test_bad_repos_labels), 2)}")
        # print(f"f1_score:{round(f1_score(preds, load_test_bad_repos_labels, average='binary'), 2)}")
        # print(f"recall_score:{round(recall_score(preds, load_test_bad_repos_labels, average='binary'), 2)}")
        # print(f"precision_score:{round(precision_score(preds, load_test_bad_repos_labels, average='binary'), 2)}")
        #
        # conf_matrix_bad.append(confusion_matrix(preds, load_test_bad_repos_labels))

    return best_model, best_preds, best_f1_score, best_conf_matrix

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

def word2_vec_representation(df, load, labels, label_mapper, nlp):
    tokenizer = LogTokenizer()
    tokenized = []
    for i in trange(0, len(df)):
        tokenized.append(nlp(df['log_message'][i]).vector)
    labels_tokenized = [label_mapper[label] for label in labels]
    return tokenized, labels_tokenized, tokenizer

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

def read_data(path):
    print("Reading data at path ", path)
    return pd.read_csv(path).drop(columns=["Unnamed: 0"])


def preprocess_data(df, scenario, verbose=True):

    if verbose:
        print("Filtering the special characters in the dataframe!")
    df['log_message'] = df['log_message'].str.replace("\<\*\>", " ")
    df['log_message'] = df['log_message'].str.replace("\[STR\]", " ")
    df['log_message'] = df['log_message'].str.replace("\[NUM\]", " ")
    if verbose:
        print("Converting the classes into required categories. Pair or triplet of (INFO, ERROR, WARNING). ")

    if scenario=="error_warning":
        df.loc[:, 'log_level'] = df.loc[:, 'log_level'].apply(lambda x: convert_error_warning(x))
    elif scenario == "info_warning":
        df.loc[:, 'log_level'] = df.loc[:, 'log_level'].apply(lambda x: convert_info_warning(x))
    elif scenario == "info_error":
        df.loc[:, 'log_level'] = df.loc[:, 'log_level'].apply(lambda x: convert_error_info(x))
    elif scenario=="info_error_warning":
        df.loc[:, 'log_level'] = df.loc[:, 'log_level'].apply(lambda x: convert_error_info_warning(x))
    else:
        print("Insert a valid scenario, one in error_warning, info_warning, info_error")
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


def run_train(dataloader, model, optimizer, f_loss, epoch, polars=None, device="cpu"):
    model.train()
    total_loss = 0
    start = time.time()
    for i, batch in enumerate(dataloader):
        load, y = batch
        if polars is not None:
            y = polars[y.numpy()]
            y = torch.autograd.Variable(y).cuda()

        if device == "gpu":
            out = model.forward(load.cuda().long())
        else:
            out = model.forward(load.long())

        if isinstance(f_loss, nn.CosineSimilarity):
            loss = (1 - f_loss(out, y)).pow(2).sum()
        else:
            if device=="gpu":
                loss = f_loss(out, y.cuda().long())
            else:
                loss = f_loss(out, y.long())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss
        elapsed = time.time() - start
        if i % 5 == 0:
            print("Epoch %d Train Step: %d / %d Loss: %f" %
                  (epoch, i, len(dataloader), loss), end='\r')

    print("Epoch %d Train Step: %d / %d Loss: %f" %
          (epoch, i, len(dataloader), loss), end='\r')
    return total_loss / len(dataloader)


def run_test(dataloader, model, optimizer, f_loss, epoch, polars=None, device="cpu"):
    model.eval()
    preds = []
    tmps = []
    scores_head1 = []
    scores_head2 = []
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
                tmps += list(tmp)
                scores_head1 += model.encoder.layers[0].self_attn.attn[:, 0, :, :].detach().cpu()
                scores_head2 += model.encoder.layers[0].self_attn.attn[:, 1, :, :].detach().cpu()
    return preds, scores_head1, scores_head2


def run_optimizer(model, train_dataloader, test_dataloader, test_dataloader_bad_repos, labels_test, labels_test_bad_repos, optimizer, n_epochs, f_loss, polars, class_weights, device):
    conf_matrix_good = []
    conf_matrix_bad = []
    best_f1_good = 0
    best_f1_bad = 0
    idx_good = 0
    idx_bad = 0
    best_model = 0
    best_preds = 0
    for epoch in range(1, 1 + n_epochs):
        print("Epoch", epoch)
        loss = run_train(train_dataloader, model, optimizer, f_loss, epoch, polars, device)
        print("Epoch %d Train Loss: %f" % (epoch, loss), " " * 30)
        start_time = time.time()

        print("----------GOOD REPOS----------")
        preds1, scores11, scores12 = run_test(test_dataloader, model, optimizer, f_loss, epoch, polars, device)
        print(f"Accuracy:{round(accuracy_score(preds1, labels_test), 2)}")
        print(f"f1_score:{round(f1_score(preds1, labels_test, average='binary'), 2)}")
        print(f"recall_score:{round(recall_score(preds1, labels_test, average='binary'), 2)}")
        print(f"precision_score:{round(precision_score(preds1, labels_test, average='binary'), 2)}")

        conf_matrix_good.append(confusion_matrix(preds1, labels_test))

        pp = confusion_matrix(preds1, labels_test)
        print(pp)
        if pp.shape[0]<3:
            if best_f1_good < f1_score(preds1, labels_test, average='binary') and pp[0][0] >0 and pp[1][1] > 0:
                best_f1_good = f1_score(preds1, labels_test, average='binary')
                idx_good = epoch-1
                best_model = model
                # torch.save(model,
                #            "/home/matilda/PycharmProjects/log_level_estimation/log_level_estimation/5_results/models/incremental/" + scenario + ".pth")
                # with open(
                #         "/home/matilda/PycharmProjects/log_level_estimation/log_level_estimation/5_results/models/incremental/" + scenario + "_label_mapper.pickle",
                #         "wb") as file:
                #     pickle.dump(label_mapper, file)
        else:
            if best_f1_good < f1_score(preds1, labels_test, average='binary') and pp[0][0] >0 and pp[1][1] > 0 and pp[2][2]:
                best_f1_good = f1_score(preds1, labels_test, average='binary')
                idx_good = epoch-1
                best_model = model
                # torch.save(model,
                #            "/home/matilda/PycharmProjects/log_level_estimation/log_level_estimation/5_results/models/incremental/" + scenario + ".pth")
                # with open(
                #         "/home/matilda/PycharmProjects/log_level_estimation/log_level_estimation/5_results/models/incremental/" + scenario + "_label_mapper.pickle",
                #         "wb") as file:
                #     pickle.dump(label_mapper, file)


        print("----------BAD REPOS----------")

        preds, scores21, scores22 = run_test(test_dataloader_bad_repos, model, optimizer, f_loss, epoch, polars, device)


        print(f"Accuracy:{round(accuracy_score(preds, labels_test_bad_repos), 2)}")
        print(f"f1_score:{round(f1_score(preds, labels_test_bad_repos, average='binary'), 2)}")
        print(f"recall_score:{round(recall_score(preds, labels_test_bad_repos, average='binary'), 2)}")
        print(f"precision_score:{round(precision_score(preds, labels_test_bad_repos, average='binary'), 2)}")

        conf_matrix_bad.append(confusion_matrix(preds, labels_test_bad_repos))
        pp = confusion_matrix(preds, labels_test_bad_repos)

        if pp.shape[0] < 3:
            if best_f1_bad < f1_score(preds, labels_test_bad_repos, average='binary') and pp[0][0] > 0 and pp[1][1] > 0:
                best_f1_bad = f1_score(preds, labels_test_bad_repos, average='binary')
                idx_bad = epoch - 1
        else:
            if best_f1_bad < f1_score(preds, labels_test_bad_repos, average='binary') and pp[0][0] > 0 and pp[1][1] > 0 and pp[2][2]:
                best_f1_bad = f1_score(preds, labels_test_bad_repos, average='binary')
                idx_bad = epoch - 1


    return best_model, preds1, preds, conf_matrix_good, conf_matrix_bad, scores11, scores12, scores21, scores22, best_f1_good, best_f1_bad, idx_good, idx_bad


def top_ranked_repos(repositories, star_repos, number_repos_good, number_bad_repos, number_validation_repos, good_bad_hypo):
    repositories= repositories.drop('index', axis=1)
    repositories = repositories.reset_index()
    repositories.columns = ["id", "repo_link"]

    if good_bad_hypo:
        top_repos = star_repos.iloc[:number_repos_good, :].repo_name
        bottom_repos = star_repos.iloc[(-1)*number_bad_repos:,:].repo_name # THIS TRAINS ON TOP repositories
    else:
        top_repos = star_repos.iloc[(-1)*number_repos_good:, :].repo_name.values
        bottom_repos = star_repos.iloc[:number_bad_repos,:].repo_name    # THIS TRAINS ON BOTTOM repos

    grepos = np.arange(number_repos_good).tolist()
    validation_repos = set(random.sample(grepos, number_validation_repos))
    train_repos = set(grepos).difference(validation_repos)

    top_ranked_indecies = []
    top_ranked_validation_indecies = []
    bottom_ranked_indecies = []
    joint = []

    for good_repos in top_repos[list(train_repos)]:
        top_ranked_indecies.append(repositories[repositories.repo_link==good_repos].id.values)
        joint.append(repositories[repositories.repo_link==good_repos].id.values)

    for good_repos in top_repos[list(validation_repos)]:
        top_ranked_validation_indecies.append(repositories[repositories.repo_link==good_repos].id.values)
        joint.append(repositories[repositories.repo_link==good_repos].id.values)

    for bad_repos in bottom_repos:
        bottom_ranked_indecies.append(repositories[repositories.repo_link==bad_repos].id.values)
        joint.append(repositories[repositories.repo_link==bad_repos].id.values)

    return np.hstack(top_ranked_indecies), np.hstack(top_ranked_validation_indecies), np.hstack(bottom_ranked_indecies), np.hstack(joint)

def create_data_loaders_baselines(load_train, labels_train, load_test,  labels_test, batch_size):
    train_data = TensorDataset(torch.tensor(load_train, dtype=torch.float32), torch.tensor(labels_train.astype(np.int32), dtype=torch.int32))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    test_data = TensorDataset(
        torch.tensor(load_test, dtype=torch.float32),
        torch.tensor(labels_test.astype(np.int32).flatten(), dtype=torch.int32))

    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return train_dataloader, test_dataloader

def evaluate(preds1, load_test_good_repos_labels, preds, load_test_bad_repos_labels, good_bad_hypo):
    fin_results = defaultdict(dict)

    print("********"*10)
    print("----------GOOD REPOS----------")

    print(f"Accuracy:{round(accuracy_score(preds1, load_test_good_repos_labels), 2)}")
    print(f"f1_score:{round(f1_score(preds1, load_test_good_repos_labels, average='binary'), 2)}")
    print(f"recall_score:{round(recall_score(preds1, load_test_good_repos_labels, average='binary'), 2)}")
    print(f"precision_score:{round(precision_score(preds1, load_test_good_repos_labels, average='binary'), 2)}")

    d = {}
    d["Accuracy"] = accuracy_score(preds1, load_test_good_repos_labels)
    d['F1_score'] = f1_score(preds1, load_test_good_repos_labels, average='binary')
    d["recall_score"] = recall_score(preds1, load_test_good_repos_labels, average='binary')
    d["precision_score"] = precision_score(preds1, load_test_good_repos_labels, average='binary')
    d["confusion_matrix"] = confusion_matrix(preds1, load_test_good_repos_labels)

    if good_bad_hypo == True:
        fin_results["good_repos"] = d
    else:
        fin_results["bad_repos"] = d

    print("----------BAD REPOS----------")

    print(f"Accuracy:{round(accuracy_score(preds, load_test_bad_repos_labels), 2)}")
    print(f"f1_score:{round(f1_score(preds, load_test_bad_repos_labels, average='binary'), 2)}")
    print(f"recall_score:{round(recall_score(preds, load_test_bad_repos_labels, average='binary'), 2)}")
    print(f"precision_score:{round(precision_score(preds, load_test_bad_repos_labels, average='binary'), 2)}")
    conf_matrix_bad.append(confusion_matrix(preds, load_test_bad_repos_labels))

    d = {}
    d["Accuracy"] = accuracy_score(preds, load_test_bad_repos_labels)
    d['F1_score'] = f1_score(preds, load_test_bad_repos_labels, average='binary')
    d["recall_score"] = recall_score(preds, load_test_bad_repos_labels, average='binary')
    d["precision_score"] = precision_score(preds, load_test_bad_repos_labels, average='binary')
    d["confusion_matrix"] = confusion_matrix(preds, load_test_bad_repos_labels)



    if good_bad_hypo == True:
        fin_results["bad_repos"] = d
    else:
        fin_results["good_repos"] = d

    return fin_results

def create_data_loaders_baselines_test( load_test,  labels_test, batch_size):
    test_data = TensorDataset(
        torch.tensor(load_test,  dtype=torch.float32),
        torch.tensor(labels_test.astype(np.int32).flatten(), dtype=torch.int32))

    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return test_dataloader


all_results = defaultdict(dict)
all_results_m1 = defaultdict(dict)
all_results_m2 = defaultdict(dict)
all_results_m3 = defaultdict(dict)

#
#
# good_bad_hypo = True
# scenario  = "info_error_warning"
# store_path = "../../5_results/models/learning_scenario1/"
# results_name = store_path + scenario + "/10_fold_sample_bin_" + str(good_bad_hypo) +  "_.pickle"
# label_mapper_name = store_path + scenario + "/label_mapper_bin_" + str(good_bad_hypo) + "_.pickle"
#
# with open(results_name, "rb") as file:
#     all_results = pickle.load(file)
#
# with open(label_mapper_name, "rb") as file:
#     label_mapper_name = pickle.load(file)
#
# store_path = "../../5_results/models/baseline/"
# results_name_m1 = store_path + scenario + "/model1_10_fold_sample_bin_" + str(good_bad_hypo) + "_.pickle"
# results_name_m2 = store_path + scenario + "/model2_10_fold_sample_bin_" + str(good_bad_hypo) + "_.pickle"
# results_name_m3 = store_path + scenario + "/model3_10_fold_sample_bin_" + str(good_bad_hypo) + "_.pickle"
# label_mapper_name = store_path + scenario + "/label_mapper_bin_" + str(good_bad_hypo) + "_.pickle"
#
# with open(results_name_m1, "rb") as file:
#     all_results_m1 = pickle.load(file)
#
# with open(results_name_m2, "rb") as file:
#     all_results_m2 = pickle.load(file)
#
# with open(results_name_m3, "rb") as file:
#     all_results_m3 = pickle.load(file)
#
# with open(label_mapper_name, "rb") as file:
#     label_mapper_name = pickle.load(file)
#
# print(all_results_m3.keys())

for seed in np.arange(1):
    print("CURRENTLY PROCESSING SEED {}".format(seed))
    PATH = "../../3_preprocessed_data/filtered_log_df_reduced.csv"
    PATH_COUNTS = "../../3_preprocessed_data/stars_repos.csv"
    learning_rate = 0.0001
    decay = 0.001
    betas = (0.9, 0.999)
    momentum = 0.9

    number_repos_good = 700
    number_bad_repos = 1
    number_validation_repos = 100


    batch_size = 2048
    pad_len = 50
    n_layers=2
    in_features=16
    out_features=16
    num_heads=2
    dropout=0.05
    max_len=50
    n_targets = 2
    device = "gpu"
    random_seed = seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    scenario = "info_error"  # ONE IN: "info_warning", "info_error", "error_warning", "info_error_warning"
    n_epochs = 50
    good_bad_hypo = True

    df = read_data(PATH)
    repositories = df['repo_link']
    df, indecies_to_preserve = preprocess_data(df, scenario)

    repositories =  repositories.loc[indecies_to_preserve]
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


    load, labels = extract_load(df)
    class_count = df.groupby("log_level").count()['log_message']
    label_mapper = {class_count.index[i]:i for i in range(len(class_count))}

    tokenized, labels_tokenized, tokenizer = tokenization_dataset(df, load, labels, label_mapper)

    assert len(tokenized) == df.shape[0], "Some data samples have been lost during tokenization. Take care of this."

    load_train = np.array(tokenized, dtype=object)[df1.loc[train_good_repos].iloc[:, 0].values]
    load_train_labels = np.array(labels_tokenized)[df1.loc[train_good_repos].iloc[:, 0].values]

    load_test_good_repos = np.array(tokenized, dtype=object)[df1.loc[validation_good_repos].iloc[:, 0].values]
    load_test_good_repos_labels = np.array(labels_tokenized)[df1.loc[validation_good_repos].iloc[:, 0].values]

    load_test_bad_repos = np.array(tokenized, dtype=object)[df1.loc[bad_repos].iloc[:, 0].values]
    load_test_bad_repos_labels = np.array(labels_tokenized)[df1.loc[bad_repos].iloc[:, 0].values]




    train_dataloader, test_dataloader_good_repos = create_data_loaders(load_train, load_train_labels, load_test_good_repos,  load_test_good_repos_labels, pad_len, batch_size)

    test_dataloader_bad_repos = create_test_data_loaders(load_test_bad_repos, load_test_bad_repos_labels, pad_len, batch_size)

    if device =="gpu":
        torch.cuda.empty_cache()


    src_vocab = tokenizer.n_words


    calculate_weights = lambda x, i: x.sum() / (len(x)*x[i])
    weights = [calculate_weights(class_count,i) for i in range(len(class_count))]
    weights /= max(weights)

    if device =="gpu":
        class_weights=torch.FloatTensor(weights).cuda()
        cross_entropoy_loss = nn.CrossEntropyLoss(weight=class_weights).cuda()
    else:
        class_weights = torch.FloatTensor(weights)
        cross_entropoy_loss = nn.CrossEntropyLoss(weight=class_weights)


    loss_f = cross_entropoy_loss

    model = NuLogsyModel(src_vocab=src_vocab, tgt_vocab=n_targets,
                         n_layers=n_layers, in_features=in_features,
                         out_features=out_features,num_heads=num_heads,
                         dropout=dropout, max_len=max_len).get_model()

    if device == "gpu":
        torch.cuda.set_device(0)
        model.cuda()

    sgd_opt = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)
    adam_opt = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=decay)
    optimizers = {"adam":adam_opt,"sgd":sgd_opt}
    optimizer = optimizers['adam']


    model, preds1, preds, conf_matrix_good, conf_matrix_bad, scores11, scores12, scores21, scores22, best_f1_good, best_f1_bad, idx_good, idx_bad = run_optimizer(model, train_dataloader, test_dataloader_good_repos, test_dataloader_bad_repos,  load_test_good_repos_labels, load_test_bad_repos_labels, optimizer, n_epochs,cross_entropoy_loss,polars=None,class_weights=weights, device=device)



    fin_results = defaultdict(dict)

    print("*******"*10)
    print("----------GOOD REPOS----------")

    print(f"Accuracy:{round(accuracy_score(preds1, load_test_good_repos_labels), 2)}")
    print(f"f1_score:{round(f1_score(preds1, load_test_good_repos_labels, average='binary'), 2)}")
    print(f"recall_score:{round(recall_score(preds1, load_test_good_repos_labels, average='binary'), 2)}")
    print(f"precision_score:{round(precision_score(preds1, load_test_good_repos_labels, average='binary'), 2)}")


    d = {}
    d["Accuracy"] = accuracy_score(preds1, load_test_good_repos_labels)
    d['F1_score_last'] = f1_score(preds1, load_test_good_repos_labels, average='binary')
    d["recall_score"] = recall_score(preds1, load_test_good_repos_labels, average='binary')
    d["precision_score"] = precision_score(preds1, load_test_good_repos_labels, average='binary')
    d["confusion_matrix"] = conf_matrix_good
    d["F1_best"] = best_f1_good
    d["F1_epoch_good"] = idx_good

    if good_bad_hypo==True:
        fin_results["good_repos"] = d
    else:
        fin_results["bad_repos"] = d

    print("----------BAD REPOS----------")

    print(f"Accuracy:{round(accuracy_score(preds, load_test_bad_repos_labels), 2)}")
    print(f"f1_score:{round(f1_score(preds, load_test_bad_repos_labels, average='binary'), 2)}")
    print(f"recall_score:{round(recall_score(preds, load_test_bad_repos_labels, average='binary'), 2)}")
    print(f"precision_score:{round(precision_score(preds, load_test_bad_repos_labels, average='binary'), 2)}")


    d = {}
    d["Accuracy"] = accuracy_score(preds, load_test_bad_repos_labels)
    d['F1_score'] = f1_score(preds, load_test_bad_repos_labels, average='binary')
    d["recall_score"] = recall_score(preds, load_test_bad_repos_labels, average='binary')
    d["precision_score"] = precision_score(preds, load_test_bad_repos_labels, average='binary')
    d["confusion_matrix"] = conf_matrix_bad
    d["F1_best"] = best_f1_bad
    d["F1_epoch_best"] = idx_bad

    if good_bad_hypo==True:
        fin_results["bad_repos"] = d
    else:
        fin_results["good_repos"] = d

    all_results[seed] = fin_results

    # nlp = spacy.load("en_core_web_sm")
    #
    # tokenized, labels_tokenized, tokenizer = word2_vec_representation(df, load, labels, label_mapper, nlp)
    #
    # assert len(tokenized) == df.shape[0], "Some data samples have been lost during tokenization. Take care of this."
    #
    # load_train = np.array(tokenized, dtype=np.float32)[df1.loc[train_good_repos].iloc[:, 0].values]
    # load_train_labels = np.array(labels_tokenized)[df1.loc[train_good_repos].iloc[:, 0].values]
    #
    # load_test_good_repos = np.array(tokenized, dtype=np.float32)[df1.loc[validation_good_repos].iloc[:, 0].values]
    # load_test_good_repos_labels = np.array(labels_tokenized)[df1.loc[validation_good_repos].iloc[:, 0].values]
    #
    # load_test_bad_repos = np.array(tokenized, dtype=np.float32)[df1.loc[bad_repos].iloc[:, 0].values]
    # load_test_bad_repos_labels = np.array(labels_tokenized)[df1.loc[bad_repos].iloc[:, 0].values]
    #
    # train_dataloader, test_dataloader_good_repos = create_data_loaders_baselines(load_train, load_train_labels,
    #                                                                              load_test_good_repos,
    #                                                                              load_test_good_repos_labels,
    #                                                                              batch_size)
    # test_dataloader_bad_repos = create_data_loaders_baselines_test(load_test_bad_repos, load_test_bad_repos_labels,
    #                                                                batch_size)
    #
    # src_vocab = tokenized[0].shape[0]
    # model1 = Baseline(n_dimension=src_vocab, n_targets=n_targets)
    #
    # if device == "gpu":
    #     torch.cuda.set_device(0)
    #     model1.cuda()
    #
    # sgd_opt = torch.optim.SGD(model1.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)
    # adam_opt = torch.optim.Adam(model1.parameters(), lr=learning_rate, betas=betas, weight_decay=decay)
    # optimizers = {"adam":adam_opt,"sgd":sgd_opt}
    # optimizer = optimizers['adam']
    #
    # n_epochs = 200
    #
    # model1, preds1, preds, conf_matrix_good, conf_matrix_bad = run_optimizer_baseline(model1, train_dataloader, test_dataloader_good_repos, test_dataloader_bad_repos,  load_test_good_repos_labels, load_test_bad_repos_labels, optimizer, n_epochs,cross_entropoy_loss,class_weights=weights, device=device)
    #
    # print("STARTED TRAINING RANDOM FOREST!")
    #
    # model2 = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, max_depth=None, max_features=int(src_vocab/3)))
    # model2.fit(load_train, load_train_labels)
    # preds_good_m2 = model2.predict(load_test_good_repos)
    # preds_bad_m2 = model2.predict(load_test_bad_repos)
    #
    # print("STARTED TRAINING SUPPORT VECTOR MACHINE!")
    # model3 = make_pipeline(StandardScaler(), SVC(gamma="auto", kernel="rbf"))
    # model3.fit(load_train, load_train_labels)
    # preds_good_m3 = model3.predict(load_test_good_repos)
    # preds_bad_m3 = model3.predict(load_test_bad_repos)
    #
    #
    # all_results_m1[seed] = evaluate(preds1, load_test_good_repos_labels, preds, load_test_bad_repos_labels, good_bad_hypo)
    # all_results_m2[seed] = evaluate(preds_good_m2, load_test_good_repos_labels, preds_bad_m2, load_test_bad_repos_labels, good_bad_hypo)
    # all_results_m3[seed] = evaluate(preds_good_m3, load_test_good_repos_labels, preds_bad_m3,
    #                                 load_test_bad_repos_labels, good_bad_hypo)
    #
    #
    # store_path = "../../5_results/models/learning_scenario1/"
    # results_name = store_path + scenario + "/10_fold_sample_bin_" + str(good_bad_hypo) +  "_.pickle"
    # label_mapper_name = store_path + scenario + "/label_mapper_bin_" + str(good_bad_hypo) + "_.pickle"
    #
    # with open(results_name, "wb") as file:
    #     pickle.dump(all_results, file)
    #
    # with open(label_mapper_name, "wb") as file:
    #     pickle.dump(label_mapper_name, file)
    #
    # store_path = "../../5_results/models/baseline/"
    # results_name_m1 = store_path + scenario + "/model1_10_fold_sample_bin_" + str(good_bad_hypo) + "_.pickle"
    # results_name_m2 = store_path + scenario + "/model2_10_fold_sample_bin_" + str(good_bad_hypo) + "_.pickle"
    # results_name_m3 = store_path + scenario + "/model3_10_fold_sample_bin_" + str(good_bad_hypo) + "_.pickle"
    # label_mapper_name = store_path + scenario + "/label_mapper_bin_" + str(good_bad_hypo) + "_.pickle"
    #
    # with open(results_name_m1, "wb") as file:
    #     pickle.dump(all_results_m1, file)
    #
    # with open(results_name_m2, "wb") as file:
    #     pickle.dump(all_results_m2, file)
    #
    # with open(results_name_m3, "wb") as file:
    #     pickle.dump(all_results_m3, file)
    #
    # with open(label_mapper_name, "wb") as file:
    #     pickle.dump(label_mapper_name, file)

from scipy.linalg import norm

def extract_gradients_input_output(model, original_test_data, ground_truth_labels, predictions_model):
    #
    # if device =="gpu":
    #     class_weights=torch.FloatTensor(weights).cuda()
    #     cross_entropoy_loss = nn.CrossEntropyLoss(weight=class_weights).cuda()
    # else:
    #     class_weights = torch.FloatTensor(weights)
    #     cross_entropoy_loss = nn.CrossEntropyLoss(weight=class_weights)
    #
    #
    # loss_f = cross_entropoy_loss

    # adam_opt = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=decay)

    df = pd.DataFrame([ground_truth_labels, predictions_model]).T
    df.columns = ["ground_truth", "prediction"]

    a = df[df.ground_truth==1]
    b = a[a.prediction==0]  # true is INFO, predicted as error
    valid_indecies = b.index

    subset_original_data = original_test_data[valid_indecies]
    subset_ground_truth_data = ground_truth_labels[valid_indecies]

    test_dataloader = create_test_data_loaders(subset_original_data, subset_ground_truth_data, pad_len, batch_size=1)

    pom = []
    batch_size = 1
    for idx, batch in enumerate(test_dataloader):

        input = batch[0].long().cuda()
        # print(input)
        # print("DOES INPUT REQUIRE GRADS ? ANSWER: ", input.requires_grad)
        # input = input.requires_grad_(True)

        # out = model.forward(input.cuda().long(), None)[0, 1].backward() !!!!!!!!!!!!!!!!!!

        y_hat = model.forward(input, explain = True, rule = "alpha2beta1")
        y_hat = y_hat[torch.arange(batch_size), y_hat.max(1)[1]]  # Choose maximizing output neuron
        y_hat = y_hat.sum()

        # Backward pass (do explanation)
        y_hat.backward()
        # explanation = x.grad


        # loss = cross_entropoy_loss(out, batch[1])
        # loss.backward()
        #
        # print(torch.autograd.grad(loss, model.src_embed[0].lut(input[0]), allow_unused=True))

        pom.append(model.src_embed[0].lut.weight.grad.cpu())

        # pom.append(model.src_embed[0].lut.weight.grad.cpu()) !!!!!!!!!!!

    # adam_opt.zero_grad()
    # loss = cross_entropoy_loss(out, class_labels)
    # loss.backward()
    #
    return b, valid_indecies, pom


# with open("/home/matilda/PycharmProjects/log_level_estimation/log_level_estimation/5_results/models/explanation/" + scenario + "_tokenizer.pth", "wb") as file:
#     pickle.dump(tokenizer, file)
# torch.save(model, "/home/matilda/PycharmProjects/log_level_estimation/log_level_estimation/5_results/models/explanation/" + scenario + ".pth")
# with open("/home/matilda/PycharmProjects/log_level_estimation/log_level_estimation/5_results/models/explanation/" + scenario + "_label_mapper.pickle", "wb") as file:
#     pickle.dump(label_mapper, file)
#



# with open("/home/matilda/PycharmProjects/log_level_estimation/log_level_estimation/5_results/models/explanation/" + scenario + "_tokenizer.pth", "rb") as file:
#     tokenizer = pickle.load(file)
# model = torch.load("/home/matilda/PycharmProjects/log_level_estimation/log_level_estimation/5_results/models/explanation/" + scenario + ".pth")
# with open("/home/matilda/PycharmProjects/log_level_estimation/log_level_estimation/5_results/models/explanation/" + scenario + "_label_mapper.pickle", "rb") as file:
#     label_mapper = pickle.load(file)

original_test_data = load_test_good_repos
ground_truth_labels = load_test_good_repos_labels
predictions_model, scores21, scores22 = run_test(test_dataloader_good_repos, model, optimizer, cross_entropoy_loss, epoch=1, polars=None, device=device)

# a, b, c, = extract_gradients_input_output(model, original_test_data, ground_truth_labels, predictions_model)

def translate_to_human_readable(tokenizer, log_msg):
    v = []
    for x in log_msg:
        if x>0:
            v.append(tokenizer.index2word[x])
    return " ".join(v)

def get_score_log_message(gradients_for_log_msg, index):
    return norm(gradients_for_log_msg[index], axis=1)

def get_word_index(tokenizer, word):
    return tokenizer.word2index[word]

# index_= 600
# q =np.arange(0, c[index_].shape[0])[np.where(get_score_log_message(c, index_), True, False)]
# print(translate_to_human_readable(tokenizer, original_test_data[index_]))
# print(translate_to_human_readable(tokenizer, q))

from pprint import pprint
def write_final_res_tof_file(final_res, fname):
    # Build the tree somehow
    with open(fname, 'wt') as out:
        pprint(final_res, stream=out)

def convert_tokenizer_to_explainer_data(load_train, model_attn, max_len):
    lista = []
    padding_vector_token = model_attn.src_embed[0].forward(torch.tensor([0]).long().cuda()).detach().cpu()
    for idx in range(load_train.shape[0]):
        tmp_list = []
        if len(load_train[idx]) < max_len:
            for j in load_train[idx]:
                tmp_list.append(model.src_embed[0].forward(torch.tensor([j]).long().cuda()).detach().cpu())
            for k in range(max_len - len(load_train[idx])):
                tmp_list.append(padding_vector_token)
        else:
            for j in range(max_len):
                tmp_list.append(model.src_embed[0].forward(torch.tensor([load_train[idx][j]]).long().cuda()).detach().cpu())
        lista.append(torch.cat(tmp_list, axis=1))
    return lista



original_test_data = load_test_good_repos
ground_truth_labels = load_test_good_repos_labels
predictions_model, scores21, scores22 = run_test(test_dataloader_good_repos, model, optimizer, cross_entropoy_loss, epoch=1, polars=None, device=device)


batch_size = 2048
n_epochs = 20

training_data = convert_tokenizer_to_explainer_data(load_train, model, max_len)
test_data = convert_tokenizer_to_explainer_data(load_test_good_repos, model, max_len)

training_data = torch.cat(training_data)
test_data = torch.cat(test_data)

train_dataloader_baseline, test_dataloader_baseline = create_data_loaders_baselines(
                                                                             training_data,
                                                                             load_train_labels,
                                                                             test_data,
                                                                             load_test_good_repos_labels,
                                                                             batch_size)




reduced_module = Baseline(in_features*max_len, n_targets, max_len, in_features)
adam_opt = torch.optim.Adam(reduced_module.parameters(), lr=learning_rate, betas=betas, weight_decay=decay)
device = "cuda"
reduced_module.to(device)
optimizer = adam_opt
reduced_module, best_preds, best_f1_score, best_conf_matrix  = run_optimizer_baseline(reduced_module, train_dataloader_baseline, test_dataloader_baseline, test_dataloader_baseline,  load_test_good_repos_labels, load_test_good_repos_labels, optimizer, n_epochs,cross_entropoy_loss,class_weights=weights, device=device)


with open("/home/matilda/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/"+ scenario +"/" + scenario + "_tokenizer.pickle", "wb") as file:
    pickle.dump(tokenizer, file)

torch.save(reduced_module, "/home/matilda/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/"+ scenario+ "/" + scenario + ".pth")

with open("/home/matilda/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/" + scenario + "/" + scenario + "_label_mapper.pickle", "wb") as file:
    pickle.dump(label_mapper, file)

with open("/home/matilda/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/" + scenario + "/" + scenario + "_training_vectors.pickle", "wb") as file:
    pickle.dump(training_data, file)

with open("/home/matilda/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/" + scenario + "/" + scenario + "_test_vectors.pickle", "wb") as file:
    pickle.dump(test_data, file)

with open("/home/matilda/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/" + scenario + "/" + scenario + "_train_labels.pickle", "wb") as file:
    pickle.dump(load_train_labels, file)

with open("/home/matilda/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/" + scenario + "/" + scenario + "_test_labels.pickle", "wb") as file:
    pickle.dump(load_test_good_repos_labels, file)

with open("/home/matilda/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/" + scenario + "/" + scenario + "_load_train.pickle", "wb") as file:
    pickle.dump(load_train, file)

with open("/home/matilda/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/" + scenario + "/" + scenario + "_load_test.pickle", "wb") as file:
    pickle.dump(load_test_good_repos, file)

with open("/home/matilda/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/" + scenario + "/" + scenario + "_original_test_data.pickle", "wb") as file:
    pickle.dump(original_test_data, file)

torch.save(test_dataloader_baseline.dataset.tensors[0], "/home/matilda/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/info_error_" + "_testdata.pth")


def process_shap_values(shap_values, original_test_data, tokenizer, valid_indecies):
    store_res = defaultdict(dict)
    for log_msg_idx, _ in enumerate(shap_values):
        vals = shap_values[log_msg_idx].reshape(-1, 16)
        words = original_test_data[log_msg_idx]
        d = defaultdict(dict)
        for word_idx in range(len(words)):
            q = {}
            q['max'] = vals[word_idx][np.abs(vals[word_idx]).argmax()]
            q['norm']  = norm(vals[word_idx])
            d[tokenizer.index2word[words[word_idx]]] = q
        d['log_message_tokenized'] = words
        d['dataset_location'] = valid_indecies[log_msg_idx]
        store_res[log_msg_idx] = d

    return store_res





df = pd.DataFrame([ground_truth_labels, predictions_model]).T
df.columns = ["ground_truth", "prediction"]

print(label_mapper)
a = df[df.ground_truth == 1]
b = a[a.prediction == 1]  # true is INFO, predicted as error
valid_indecies = b.index


number_samples_to_train = 30000
number_samples_to_eval = 100
class_ = 1

valid_indecies = np.random.choice(valid_indecies, number_samples_to_eval)

perm = torch.randperm(train_dataloader_baseline.dataset.tensors[0].size(0))
idx = perm[:number_samples_to_train]
shap_train_samples = train_dataloader_baseline.dataset.tensors[0][idx]

print("I have selected the samples!")
e = shap.DeepExplainer(reduced_module, shap_train_samples.cuda())
print("Calculating SHAP values!")
shap_values = e.shap_values(test_dataloader_baseline.dataset.tensors[0][valid_indecies].cuda())
print("Plotting results for class {}".format(class_))
final_res = process_shap_values(shap_values[class_], original_test_data[valid_indecies], tokenizer, valid_indecies)


fname = "/home/matilda/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/info_error/true_info_pred_info.txt"
write_final_res_tof_file(final_res, fname)

def translate_dict_to_list(final_res):
    experiment = []
    for key in final_res.keys():
        words_ = []
        meta_info = []
        for key2 in final_res[key].keys():
            if isinstance(final_res[key][key2], dict):
                for key3 in final_res[key][key2].keys():
                    words_.append(final_res[key][key2][key3])
            else:
                meta_info.append(final_res[key][key2])

        experiment.append((words_, meta_info))

    return experiment


df = pd.DataFrame([ground_truth_labels, predictions_model]).T
df.columns = ["ground_truth", "prediction"]
df.to_csv("/home/matilda/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/info_error/prediction.csv", index=False)

print(label_mapper)
a = df[df.ground_truth == 0]
b = a[a.prediction == 0]  # true is INFO, predicted as error
valid_indecies = b.index


number_samples_to_train = 30000
number_samples_to_eval = 100
class_ = 1

valid_indecies = np.random.choice(valid_indecies, number_samples_to_eval)

perm = torch.randperm(train_dataloader_baseline.dataset.tensors[0].size(0))
idx = perm[:number_samples_to_train]
shap_train_samples = train_dataloader_baseline.dataset.tensors[0][idx]



print("I have selected the samples!")
e = shap.DeepExplainer(reduced_module, shap_train_samples.cuda())
print("Calculating SHAP values!")
shap_values = e.shap_values(test_dataloader_baseline.dataset.tensors[0][valid_indecies].cuda())
print("Plotting results for class {}".format(class_))
final_res1 = process_shap_values(shap_values[class_], original_test_data[valid_indecies], tokenizer, valid_indecies)


fname = "/home/matilda/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/info_error/true_error_pred_error.txt"
write_final_res_tof_file(final_res1, fname)



torch.save(reduced_module, "/home/matilda/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/info_error/SHAP_neural_network.pth")
torch.save(shap_train_samples, "/home/matilda/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/info_error/SHAP_training_data.pth")


vectors = {}
for x in tokenizer.index2word.keys():
    vectors[x] = model.src_embed[0].forward(torch.tensor(x).long().cuda()).detach().cpu().numpy()
    if x%100 == 0:
        print(model.src_embed[0].forward(torch.tensor(x).long().cuda()).detach().cpu().numpy())


with open("/home/matilda/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/info_error/embeddings.pickle", "wb") as file:
    pickle.dump(vectors, file)