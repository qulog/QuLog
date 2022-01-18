import pandas as pd
import  pickle
from copy import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from os.path import expanduser
import matplotlib as mpl
import matplotlib.font_manager as font_manager
mpl.use("TkAgg")
fontpath = expanduser("~/.local/share/fonts/LinLibertine_R.otf")
prop = font_manager.FontProperties(fname=fontpath)
mpl.rcParams['font.family'] = prop.get_name()
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.size'] = 16


import numpy as np

import sys
sys.path.append("classes")


from classes import *


def plot_log_message(log_message_stats, tokenizer):
    log_msg_order = log_message_stats["log_message_tokenized"]
    log_msg_order = log_msg_order[1:-2]
    print(log_msg_order)



    log_message_stats.pop("dataset_location")
    log_message_stats.pop("log_message_tokenized")

    lista_indecies = []
    # print(log_message_stats.keys())

    for idx, x in enumerate(log_msg_order):
        lista_indecies.append(((idx+1)*1.1, 0.5))

    plt.xlim(lista_indecies[0][0], lista_indecies[-1][0]+0.5)

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

        plt.text(lista_indecies[idx][0], lista_indecies[idx][1], tokenizer.index2word[log_msg_order[idx]], size=25, rotation=0, bbox=dict(boxstyle="square", facecolor=color, alpha = intensity[tokenizer.index2word[log_msg_order[idx]]]))
            # ha = "right", va = "top",
    plt.axis("off")
        #

scenario = "info_error"

with open("/home/na/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/"+ scenario +"/" + scenario + "_tokenizer.pickle", "rb") as file:
    tokenizer = pickle.load(file)

test_data = pd.read_csv("/home/na/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/info_error/error_error_modifiedAUGUMENTED.csv")

with open("/home/na/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/info_error/info_error_modified.pickle", "rb") as file:
    res_modifed = pickle.load(file)

with open("/home/na/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/info_error/info_error_original.pickle", "rb") as file:


    res_original = pickle.load(file)


index = 62
plot_log_message(copy(res_modifed[index]), tokenizer)
plt.figure(2)
plot_log_message(copy(res_original[index]), tokenizer)

print("ORIGINAL ", copy(res_original[index]))
print("MODIFIED ", copy(res_modifed[index]))
print("PREDICTIONS ORIGINAL: ", test_data.original_preds.iloc[index])
print("PREDICTIONS MODIFIED: ", test_data.modified_preds.iloc[index])

a = test_data[test_data.original_preds==0]
a = a.iloc[:, 1:]
b = a[a.modified_preds==1]

valid_keys = b.index


def get_ranked_scores(log_message_stats, tokenizer):
    log_msg_order = log_message_stats["log_message_tokenized"]
    log_message_stats.pop("dataset_location")
    log_message_stats.pop("log_message_tokenized")
    intensity = []

    for x in log_msg_order:
        intensity.append(log_message_stats[tokenizer.index2word[x]]['norm'])
    return np.flip(np.argsort(intensity))

def one_error(real_pos, predicted, k):
    if real_pos in predicted[:k]:
        return 0.0
    else:
        return 1.0

lita = []
for k in [1, 2, 3, 4, 5, 6, 7]:
    k = k
    p_atk = []
    for key in valid_keys:
        print("----------------"*10)
        p_atk.append(one_error(test_data.location_changed[key], get_ranked_scores(copy(res_modifed[key]), tokenizer).tolist(), k=k))
    lita.append(np.mean(p_atk))

plt.figure(3)
plt.scatter([1, 2, 3, 4, 5, 6, 7], lita, c="black", s=60)
plt.ylabel("One error")
plt.xlabel("Top-k ranks")
plt.vlines(test_data.original_log_message.apply(lambda x: len(x.rsplit())).median(), ymin=np.min(lita), ymax=np.max(lita), color="r", linestyles="dotted", label="median length log messages")
plt.xlim(0.5, 7.5)
plt.axis("on")
plt.legend()
print(lita)