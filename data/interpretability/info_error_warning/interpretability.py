import torch
import shap
import pickle

import torch.nn as nn

import time

from copy import copy

import pandas as pd
import numpy as np
import sys
sys.path.append("classes")


from classes import *

from collections import defaultdict
from scipy.linalg import norm

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import matplotlib.pyplot as plt


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


def run_test_baseline(dataloader, model):
    model.eval()
    preds = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            load, y = batch
            out = model.forward(load.cuda())
            tmp = out.detach().cpu().numpy()
            preds += list(np.argmax(tmp, axis=1))
    return preds

def run_optimizer_baseline(model, train_dataloader, test_dataloader_good_repos, test_dataloader_bad_repos,  load_test_good_repos_labels, load_test_bad_repos_labels, optimizer, n_epochs,cross_entropoy_loss,class_weights, device):
    conf_matrix_good = []

    best_f1_score = 0
    best_conf_matrix = []
    best_model = []
    best_preds = []

    for epoch in range(1, 1 + n_epochs):
        loss = run_train_baseline(train_dataloader, model, optimizer, cross_entropoy_loss, epoch, device=device)

        print("Epoch %d Train Loss: %f" % (epoch, loss), " " * 30)

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

scenario = "info_error_warning"

df = pd.read_csv("/home/na/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/info_error_warning/prediction.csv")
shap_train_samples = torch.load("/home/na/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/info_error_warning/SHAP_training_data.pth")
reduced_module = torch.load("/home/na/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/info_error_warning/SHAP_neural_network.pth")

with open("/home/na/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/"+ scenario +"/" + scenario + "_tokenizer.pickle", "rb") as file:
    tokenizer = pickle.load(file)

with open("/home/na/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/" + scenario + "/" + scenario + "_label_mapper.pickle", "rb") as file:
    label_mapper = pickle.load(file)

with open("/home/na/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/" + scenario + "/" + scenario + "_original_test_data.pickle", "rb") as file:
    original_test_data = pickle.load(file)

with open("/home/na/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/info_error_warning/embeddings.pickle", "rb") as file:
    vectors = pickle.load(file)


from pprint import pprint
def write_final_res_tof_file(final_res, fname):
    # Build the tree somehow
    with open(fname, 'wt') as out:
        pprint(final_res, stream=out)

test_dataloader_baseline = torch.load("/home/na/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/info_error_warning_" + "_testdata.pth")

def plot_log_message(log_message_stats, tokenizer):
    log_msg_order = log_message_stats["log_message_tokenized"]
    print(log_msg_order)



    log_message_stats.pop("dataset_location")
    log_message_stats.pop("log_message_tokenized")

    lista_indecies = []
    # print(log_message_stats.keys())
    for idx, x in enumerate(log_msg_order):
        lista_indecies.append((idx*5+1, 0.5))

    plt.xlim(lista_indecies[0][0]-10, lista_indecies[-1][0]+10)

    print(lista_indecies)
    intensity = {}
    sum = 0
    for x in log_msg_order:
        intensity[tokenizer.index2word[x]] = log_message_stats[tokenizer.index2word[x]]['norm']
        sum+=intensity[tokenizer.index2word[x]]

    for key in intensity.keys():
        intensity[key] = intensity[key]/sum

    print(intensity)

    for idx, _ in enumerate(log_msg_order):
        print(idx)
        if log_message_stats[tokenizer.index2word[log_msg_order[idx]]]['max'] <= 0:
            color = "red"
        else:
            color = "blue"

        plt.text(lista_indecies[idx][0], lista_indecies[idx][1], tokenizer.index2word[log_msg_order[idx]], size=15, rotation=0, bbox=dict(boxstyle="square", facecolor=color, alpha = intensity[tokenizer.index2word[log_msg_order[idx]]]))
            # ha = "right", va = "top",
    plt.axis("off")
        #

print(label_mapper)
a = df[df.ground_truth == 2]
b = a[a.prediction == 2]  # true is INFO, predicted as error
c = df[df.ground_truth == 1]  # true is INFO, predicted as error
c = c[df.prediction == 1]  # true is INFO, predicted as error
d = df[df.ground_truth == 0]  # true is INFO, predicted as error
d = d[df.prediction == 0]  # true is INFO, predicted as error

c = pd.concat([b.iloc[:33, :], c.iloc[1000:1033, :], d.iloc[300:334, :]], axis=0)
valid_indecies = c.index
class_ = 1


# np.random.seed(0)
# valid_indecies = np.random.choice(valid_indecies, 100)
# valid_indecies = valid_indecies[400:500]
# valid_indecies = valid_indecies[:5]


# print("I have selected the samples!")
# e = shap.DeepExplainer(reduced_module.cuda(), shap_train_samples.cuda())
# print("Calculating SHAP values!")
# shap_values = e.shap_values(test_dataloader_baseline[valid_indecies].cuda())
# print("Plotting results for class {}".format(class_))
# final_res = process_shap_values(shap_values[class_], original_test_data[valid_indecies], tokenizer, valid_indecies)
#
#
# final_res1 = copy(final_res)

# plot_log_message(copy(final_res1[1]), tokenizer)




def create_data_loaders_baselines_test(load_test, labels_test, batch_size):
    test_data = TensorDataset(
        torch.tensor(load_test,  dtype=torch.float32),
        torch.tensor(labels_test.astype(np.int32).flatten(), dtype=torch.int32))

    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return test_dataloader


def convert_tokenizer_to_explainer_data(load_train, vectors, max_len):
    lista = []
    padding_vector_token = torch.from_numpy(vectors[0])
    for idx in range(load_train.shape[0]):
        tmp_list = []
        if len(load_train[idx]) < max_len:

            for j in load_train[idx]:
                tmp_list.append(torch.from_numpy(vectors[j]))
                print("size {}".format(vectors[j]))
            for k in range(max_len - len(load_train[idx])):
                tmp_list.append(padding_vector_token)
        else:
            for j in range(max_len):
                tmp_list.append(torch.from_numpy(vectors[load_train[idx][j]]))
        print(torch.cat(tmp_list, axis=0).shape)
        lista.append(torch.cat(tmp_list, axis=0))
    return lista


batch_size = 1
max_len = 50

def translate_log_messages_index_to_word(tokenized, tokenizer):
    dataset = []
    for x in tokenized:
        log_msg = []
        for j in x:
            log_msg.append(tokenizer.index2word[j])
        dataset.append(" ".join(log_msg))
    return dataset

def translate_log_messages_word_to_index(tokenized, tokenizer):
    dataset = []
    for x in tokenized:
        log_msg = []
        for j in x.rsplit(" "):
            log_msg.append(tokenizer.word2index[j])
        dataset.append(np.array(log_msg))
    return dataset

q = c.loc[valid_indecies]
q = q.reset_index()
q = q.iloc[:, 1:]


translated_log_messages = translate_log_messages_index_to_word(original_test_data[valid_indecies], tokenizer)
df_tokenized = pd.DataFrame(translated_log_messages)
df_tokenized = pd.concat([df_tokenized, df_tokenized], axis=1)
df_tokenized["word_changed"] = np.zeros(df_tokenized.shape[0])
df_tokenized["word_inserted"] = np.zeros(df_tokenized.shape[0])
df_tokenized["index_word_changed"] = np.zeros(df_tokenized.shape[0])
df_tokenized["ground_truth_changed"] = np.zeros(df_tokenized.shape[0])
df_tokenized = pd.concat([df_tokenized, q], axis=1)


TO_GENERATE = False



if TO_GENERATE == True:
    df_tokenized.columns = ["original_log_message", "modified_log_message", "word_changed","word_inserted", "location_changed", "ground_truth_changed", 'ground_truth',  'prediction']
    df_tokenized.to_csv("/home/na/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/info_error_warning/info_error_warning_some_thing.csv", index=False)
else:
    class_ = 1

    test_data = pd.read_csv("/home/na/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/info_error_warning/info_error_warning_some_thing.csv")
    test_data = test_data[test_data.ground_truth_changed != test_data.ground_truth]

    to_translate = test_data.modified_log_message.values

    load_test_data = np.array(translate_log_messages_word_to_index(to_translate, tokenizer), dtype="object")

    load_test = convert_tokenizer_to_explainer_data(load_test_data, vectors, max_len)
    load_test_artificial_truth = np.ones(len(load_test))
    test_dataloader_good_repos = create_data_loaders_baselines_test(torch.vstack(load_test), load_test_artificial_truth, batch_size)
    preds_modified = run_test_baseline(test_dataloader_good_repos, reduced_module.cuda())

    def calc_shap(reduced_module, shap_train_samples, test_data_baseline, original_test_data, class_, valid_indecies):
        print("I have selected the samples!")
        e = shap.DeepExplainer(reduced_module.cuda(), shap_train_samples.cuda())
        print("Calculating SHAP values!")
        shap_values = e.shap_values(test_data_baseline.cuda())
        print("Plotting results for class {}".format(class_))
        final_res = process_shap_values(shap_values[class_], original_test_data, tokenizer, valid_indecies)
        return final_res

    valid_indecies = np.arange(load_test_data.shape[0])
    test_data_baseline = test_dataloader_good_repos.dataset.tensors[0]
    res_modifed = calc_shap(reduced_module, shap_train_samples, test_data_baseline, load_test_data,  class_, valid_indecies)

    with open("/home/na/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/info_error_warning/info_error_warning_modified.pickle", "wb") as file:
        pickle.dump(res_modifed, file)

    to_translate_original = test_data.original_log_message.values
    load_test_data_original = np.array(translate_log_messages_word_to_index(to_translate_original, tokenizer), dtype="object")

    load_test_original = convert_tokenizer_to_explainer_data(load_test_data_original, vectors, max_len)
    load_test_artificial_truth_original = np.ones(len(load_test_original))
    test_dataloader_good_repos_original = create_data_loaders_baselines_test(torch.vstack(load_test_original), load_test_artificial_truth_original, batch_size)
    preds_original = run_test_baseline(test_dataloader_good_repos_original, reduced_module.cuda())


    valid_indecies = np.arange(load_test_data_original.shape[0])
    test_data_baseline_orignal = test_dataloader_good_repos_original.dataset.tensors[0]

    res_original = calc_shap(reduced_module, shap_train_samples, test_data_baseline_orignal, load_test_data_original,  class_, valid_indecies)

    with open("/home/na/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/info_error_warning/info_error_warning_original.pickle", "wb") as file:
        pickle.dump(res_original, file)