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

scenario = "info_error_warning"

with open("/home/na/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/"+ scenario +"/" + scenario + "_tokenizer.pickle", "rb") as file:
    tokenizer = pickle.load(file)

test_data = pd.read_csv("/home/na/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/info_error_warning/info_error_warning_some_thing.csv")
test_data = test_data[test_data.ground_truth_changed != test_data.ground_truth]
test_data = test_data.reset_index()

with open("/home/na/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/info_error_warning/info_error_warning_modified.pickle", "rb") as file:
    res_modifed = pickle.load(file)

with open("/home/na/PycharmProjects/log_level_estimation/log_level_estimation/5_results/interpretability/info_error_warning/info_error_warning_original.pickle", "rb") as file:
    res_original = pickle.load(file)


index = 5
plot_log_message(copy(res_modifed[index]), tokenizer)
plt.figure(2)
plot_log_message(copy(res_original[index]), tokenizer)

print("ORIGINAL ", copy(res_original[index]))
print("MODIFIED ", copy(res_modifed[index]))
print("PREDICTIONS CHANGED: ", test_data.ground_truth_changed.iloc[index])
print("PREDICTIONS ground_truth: ", test_data.ground_truth.iloc[index])

a = test_data[test_data.ground_truth==0]
a = a.iloc[:, 1:]
b = a[a.ground_truth_changed==1]

valid_keys = test_data.index ### TAKE CARE OF THIS


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
plt.scatter([1, 2, 3, 4, 5, 6, 7], [0.2413793103448276, 0.08620689655172414, 0.05172413793103448, 0.034482758620689655, 0.0, 0.0, 0.0], c="black", s=60 , label="info error")
plt.scatter([1, 2, 3, 4, 5, 6, 7], lita, s=60, c="grey", label="info warning error")
plt.scatter([1, 2, 3, 4, 5, 6, 7], [0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0], c="blue", s=60 , label="random assignment")
plt.ylabel("One error")
plt.xlabel("Top-k ranks")
plt.vlines(test_data.original_log_message.apply(lambda x: len(x.rsplit())).median(), ymin=np.min(lita), ymax=np.max(lita), color="r", linestyles="dotted", label="median length log messages")
plt.xlim(0.5, 7.5)
plt.axis("on")
plt.legend()