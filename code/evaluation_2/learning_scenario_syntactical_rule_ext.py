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
from sklearn.model_selection import train_test_split

sys.path.append("classes")
from loss_functions import NuLogsyLossCompute
from model import *
from networks import *
from tokenizer import *
from data_loader import *
from prototype import get_prototypes
import torch.nn.functional as F
import spacy



from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

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
    load = df['log_messages'].values
    labels = df['label'].values
    return load, labels

def tokenization_dataset(df, load, labels, label_mapper):
    tokenizer = LogTokenizer()
    tokenized = []
    for i in trange(0, len(df)):
        tokenized.append(np.array(tokenizer.tokenize(df['log_messages'][i])))
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
        if device=="gpu":
            out = model.forward(load.cuda().long(), None)
        else:
            out = model.forward(load.long(), None)
        if isinstance(f_loss, nn.CosineSimilarity):
            loss = (1 - f_loss(out, y)).pow(2).sum()
        else:
            if device == "gpu":
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
    # tmps = []
    scores_head1 = []
    scores_head2 = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            load, y = batch
            if device=="gpu":
                out = model.forward(load.cuda().long(), None)
            else:
                out = model.forward(load.long(), None)

            if device=="cuda":
                tmp = out.detach().cpu().numpy()
            else:
                tmp = out.detach().cpu().numpy()
            preds += list(np.argmax(tmp, axis=1))

            # tmps += list(tmp)
            # scores_head1 += model.encoder.layers[0].self_attn.attn[:, 0, :, :].detach().cpu()
            # scores_head2 += model.encoder.layers[0].self_attn.attn[:, 1, :, :].detach().cpu()
    return preds, scores_head1, scores_head2


def run_optimizer(model, train_dataloader, test_dataloader, labels_test, optimizer, n_epochs, f_loss, polars, class_weights, device):
    conf_matrix_good = []
    best_f1 = 0
    best_preds = 0
    epoch_idx = 0
    for epoch in range(1, 1 + n_epochs):

        loss = run_train(train_dataloader, model, optimizer, f_loss, epoch, polars, device=device)

        preds1, scores11, scores12 = run_test(test_dataloader, model, optimizer, f_loss, epoch, polars, device=device)
        # if epoch % 5 == 0:
        print("Epoch", epoch)
        print("Epoch %d Train Loss: %f" % (epoch, loss), " " * 30)
        print("----------GOOD REPOS----------")
        print(f"Accuracy:{round(accuracy_score(preds1, labels_test), 2)}")
        print(f"f1_score:{round(f1_score(preds1, labels_test, average='binary'), 2)}")
        print(f"recall_score:{round(recall_score(preds1, labels_test, average='binary'), 2)}")
        print(f"precision_score:{round(precision_score(preds1, labels_test, average='binary'), 2)}")
        print(confusion_matrix(preds1, labels_test))
        f1sc = f1_score(preds1, labels_test, average='binary')

        if f1sc > best_f1:
            best_f1 = f1sc
            epoch_idx = epoch-1
            best_preds = preds1

        conf_matrix_good.append(confusion_matrix(preds1, labels_test))
    return model, best_preds, conf_matrix_good, best_f1, epoch_idx

def merge_data(list_paths):
    tmp_list = []
    list_indecies_test = []
    for idx, fname in enumerate(list_paths):
        tmp_df = pd.read_csv(fname)
        tmp_df["version"] = np.ones(tmp_df.shape[0])*(idx+1)
        if idx==0:
            list_indecies_test.append(tmp_df.shape[0])
        else:
            list_indecies_test.append(list_indecies_test[-1] + tmp_df.shape[0])
        tmp_list.append(tmp_df)
    return pd.concat(tmp_list, axis=0), list_indecies_test

def split_train_test_dfs(df, train_versions, test_versions):
    train_dfs = []
    for version in train_versions:
        train_dfs.append(df[df.version==version].index)
    test_dfs = []
    for version in test_versions:
        test_dfs.append(df[df.version==version].index)
    return np.hstack(train_dfs), np.hstack(test_dfs)


def select_relevant_logs(repo, top_ranked_repos):
    if repo in top_ranked_repos:
        return True
    else:
        return False

def word2_vec_representation(df, load, labels, label_mapper, nlp):
    tokenizer = LogTokenizer()
    tokenized = []
    for i in trange(0, len(df)):
        tokenized.append(nlp(df['log_messages'][i]).vector)
    labels_tokenized = [label_mapper[label] for label in labels]
    return tokenized, labels_tokenized, tokenizer


from collections import defaultdict

def evaluate(preds1, load_test_good_repos_labels):
    print("********"*10)
    print("----------GOOD REPOS----------")

    print(f"Accuracy:{round(accuracy_score(preds1, load_test_good_repos_labels), 2)}")
    print(f"f1_score:{round(f1_score(preds1, load_test_good_repos_labels, average='binary'), 2)}")
    print(f"recall_score:{round(recall_score(preds1, load_test_good_repos_labels, average='binary'), 2)}")
    print(f"precision_score:{round(precision_score(preds1, load_test_good_repos_labels, average='binary'), 2)}")
    print(confusion_matrix(preds1, load_test_good_repos_labels))

    d = {}
    d["Accuracy"] = accuracy_score(preds1, load_test_good_repos_labels)
    d['F1_score'] = f1_score(preds1, load_test_good_repos_labels, average='binary')
    d["recall_score"] = recall_score(preds1, load_test_good_repos_labels, average='binary')
    d["precision_score"] = precision_score(preds1, load_test_good_repos_labels, average='binary')
    d["confusion_matrix"] = confusion_matrix(preds1, load_test_good_repos_labels)

    return d


nlp = spacy.load("en_core_web_sm")

def extract_rules_train(data):
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
    train_data = train_data[train_data.label==0]
    train_rules = extract_rules_train(train_data)
    test_prediction = predict(test_data, train_rules)
    test_data["syntactic_prediction"] = test_prediction
    return test_labels, np.array(test_prediction), test_data


####### USE TO GENERATE TOP RANKED REPOS
# PATH_v1= "../../3_preprocessed_data/learning_scenario_3/quality_training_dataset_3.csv"
# PATH_filterd = "../../3_preprocessed_data/filtered_log_df.csv"
# PATH_ranks = "../../3_preprocessed_data/stars_repos.csv"
#
# df = pd.read_csv(PATH_v1)
# ranks = pd.read_csv(PATH_ranks)
# top_ranked_repos = ranks.repo_name.iloc[:100].values
#
# filtered_logs = pd.read_csv(PATH_filterd).repo_link.loc[df.key]
# filtered_logs = filtered_logs.reset_index()
# filtered_logs.columns = ["key", "repo_link"]
# df = pd.merge(df, filtered_logs, on="key", how="inner")
#
# df['to_drop'] = df.repo_link.apply(lambda x: select_relevant_logs(x, top_ranked_repos))
# df = df[df.to_drop==True]
#
#
#
#
#

# scenario = "info_error_warning" # ONE IN: "info_warning", "info_error", "error_warning", "info_error_warning"



def preprocess_log_msg(x):
    x = x.replace(">", "")
    x = x.replace("<", "")
    x = x.replace("*", "")
    x = x.replace("[", "")
    x = x.replace("]", "")
    return x

###################  DO NOT UNCOMMENT THIS df1.to_csv("/home/matilda/PycharmProjects/log_level_estimation/log_level_estimation/3_preprocessed_data/learning_scenario_3/top100_repositories.csv")

# def sytactcial_rule_prediction():
path_store_test_res = "../../5_results/sytactic_evaluation/stored_predictions/"
path_store_test_res_dictionary = "../../5_results/sytactic_evaluation/stored_predictions/"
# all_results = defaultdict(dict)
with open(path_store_test_res_dictionary + "dict_results_.pickle", "rb") as file:
    all_results = pickle.load(file)

print(all_results.keys())

for seed in range(2, 30):

    df = pd.read_csv("../../3_preprocessed_data/learning_scenario_3/top100_processed_manual_fix_processed.csv")
    df = df.loc[:, ["log_messages", "len", "valid_messages", "label"]]

    df = df.reset_index()
    df = df.drop(["index"], axis=1)

    print("Currently exploting the classes ", np.unique(df.label))

    load, labels = extract_load(df)
    class_count = df.groupby("label").count()['len']
    label_mapper = {class_count.index[i]: i for i in range(len(class_count))}
    tokenized, labels_tokenized, tokenizer = tokenization_dataset(df, load, labels, label_mapper)
    assert len(tokenized) == df.shape[0], "Some data samples have been lost during tokenization. Take care of this."

    from sklearn.model_selection import StratifiedShuffleSplit

    # seed = 10
    stSampling = StratifiedShuffleSplit(1, test_size=0.3, random_state=seed)

    load_train1 = np.array(tokenized, dtype=object)
    load_train_labels1 = np.array(labels_tokenized)

    df1 = copy.copy(df)
    for train_index, test_index in stSampling.split(load_train1, load_train_labels1):

        learning_rate = 0.0001
        device = "gpu"
        decay = 0.001
        betas = (0.8, 0.999)
        momentum = 0.8
        n_epochs = 200
        batch_size = 64
        pad_len = max_len = 32
        n_layers = 8
        in_features = 16
        out_features = 16
        num_heads = 8
        dropout = 0.05

        n_classes = 2
        random_seed = seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)


        load_train = load_train1[train_index]
        load_train_labels = load_train_labels1[train_index]
        load_test = load_train1[test_index]
        load_test_labels = load_train_labels1[test_index]


        train_dataloader, test_dataloader = create_data_loaders(load_train, load_train_labels, load_test,  load_test_labels, pad_len, batch_size)


        if device == "gpu":
            torch.cuda.empty_cache()

        src_vocab = tokenizer.n_words

        calculate_weights = lambda x, i: x.sum() / (len(x)*x[i])
        weights = [calculate_weights(class_count,i) for i in range(len(class_count))]

        # weights /= max(weights)

        weights = np.array([1., 1]) # weights = np.array([2.5, 1])
        # weights = max(weights) / weights
        # tmp = weights[1]
        # weights[1] = weights[0]
        # weights[0] = tmp

        if device == "gpu":
            class_weights=torch.FloatTensor(weights).cuda()
            cross_entropoy_loss = nn.CrossEntropyLoss(weight=class_weights).cuda()
        else:
            class_weights = torch.FloatTensor(weights)
            cross_entropoy_loss = nn.CrossEntropyLoss(weight=class_weights)


        loss_f = cross_entropoy_loss
        #
        model = NuLogsyModel(src_vocab=src_vocab, tgt_vocab=n_classes,
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

        model, preds, conf_matrix, best_f1, epoch_idx = run_optimizer(model, train_dataloader, test_dataloader, load_test_labels, optimizer, n_epochs, cross_entropoy_loss,polars=None,class_weights=weights, device=device)



        tokenized, labels_tokenized, tokenizer = word2_vec_representation(df, load, labels, label_mapper, nlp)
        assert len(tokenized) == df.shape[0], "Some data samples have been lost during tokenization. Take care of this."

        load_train = np.array(tokenized, dtype=np.float32)[train_index]
        load_train_labels = np.array(labels_tokenized)[train_index]

        load_test_good_repos = np.array(tokenized, dtype=np.float32)[test_index]
        load_test_good_repos_labels = np.array(labels_tokenized)[test_index]

        print("STARTED TRAINING RANDOM FOREST!")

        model2 = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, max_depth=None, max_features=int(tokenized[0].shape[0]/ 3)))
        model2.fit(load_train, load_train_labels)
        preds_good_m2 = model2.predict(load_test_good_repos)


        print("STARTED TRAINING SUPPORT VECTOR MACHINE!")
        model3 = make_pipeline(StandardScaler(), SVC(gamma="auto", kernel="rbf"))
        model3.fit(load_train, load_train_labels)
        preds_good_m3 = model3.predict(load_test_good_repos)


        df_train = df.loc[train_index]
        df_test = df.loc[test_index]

        load_test_labels, synct_pred, test_data = syntactic_rule(df_train, df_test, load_test_labels)

        res_seed = defaultdict(dict)

        att = evaluate(preds, load_test_labels)
        att['best_f1'] = best_f1
        att['best_f1_epoch_idx'] = epoch_idx
        att['f1_conf_matricies'] = conf_matrix

        rf = evaluate(preds_good_m2, load_test_labels)
        svm = evaluate(preds_good_m3, load_test_labels)
        synt = evaluate(synct_pred, load_test_labels)

        res_seed["attention"] = att
        res_seed["RandomForest"] = rf
        res_seed["Svm"] = svm
        res_seed["SyntacticRules"] = synt

        all_results[seed] = res_seed

        df_test["labels_attn"] = preds
        df_test["labels_RF"] = preds_good_m2
        df_test["labels_SVM"] = preds_good_m3
        df_test.to_csv(path_store_test_res+"results_"+str(seed), index=False)

        with open(path_store_test_res_dictionary+"dict_results_.pickle", "wb") as file:
            pickle.dump(all_results, file)

# models_location = "../../5_results/models/learning_scenario_2/"
# model_name = models_location + "/" +  scenario + "/info_error_warning_train_1_test_2345.pth"
# tokenizer_name = models_location + scenario + "/info_error_warning_train_1_test_2345_tokenizer.pickle"
# conf_matrix_name = models_location + scenario + "/info_error_warning_train_1_test_2345_confusion_matrix.pickle"
# label_mapper_name = models_location + scenario + "/mapper_train_1_test_2345.pickle"

# torch.save(model, model_name)
# with open(tokenizer_name, "wb") as file:
#     pickle.dump(tokenizer, file)
# with open(conf_matrix_name, "wb") as file:
#     pickle.dump(conf_matrix, file)
# with open(label_mapper_name, "wb") as file:
#     pickle.dump(label_mapper, file)