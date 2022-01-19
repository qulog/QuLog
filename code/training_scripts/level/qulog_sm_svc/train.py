import os
import spacy
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils import *


#################################################
##### Preprocessing
#################################################

def get_small_web_embeddings(vec):
    embeddings = []
    for doc in vec:
        embeddings.append(doc.vector)
    return embeddings

def preprocessing(features, targets, nlp, label2id, buffer=True):
    # Buffering this to prevent recalculation
    token_embeddings_file = "./.level_small_sm_token_embeddings.pkl"
    if buffer and os.path.isfile(token_embeddings_file):
        print("Loading token embeddings from existing buffer")
        token_embeddings = load_pickle(token_embeddings_file)
    else:
        docs = get_docs(nlp, features)
        token_embeddings = get_small_web_embeddings(docs)
        if buffer:
            store_pickle(token_embeddings, token_embeddings_file)
    token_embeddings = np.asarray(token_embeddings)


    target_ids_file = "./.level_target_ids.pkl"
    if buffer and os.path.isfile(target_ids_file):
        print("Loading target IDs from existing buffer")
        target_ids = load_pickle(target_ids_file)
    else:
        target_ids = [label2id[t] for t in targets]
        if buffer:
            store_pickle(target_ids, target_ids_file)
    target_ids = np.asarray(target_ids)

    return token_embeddings, target_ids


def sub_sample(data, num):
    levels = data.log_level.unique()
    sub_sampled = []
    for l in levels:
        l_data = data[data["log_level"] == l]
        n = num
        if len(l_data) < n:
            sub_sampled.append(l_data)
            n = len(l_data)
        else:
            sub_sampled.append(l_data.sample(n=n))
        print("Adding {} logs with level {}".format(n, l))
    return pd.concat(sub_sampled)

    
label2id = {"info": 0, "debug": 0, "trace": 0, "warning": 1, "error":1, "exception": 1, "critical": 1}
id2label = {0: "info", 1: "error"}

# Adjust if needed
data = pd.read_csv("./training_data_log_level_pred.csv")  # here add the log level dataset

data = data[data['log_level'].isin(label2id)]
data["log_level"] = data["log_level"].apply(lambda x: "warning" if x == "warn" else x)
data = sub_sample(data, 5000)

print("Dataset size: {}".format(len(data)))
features, targets = data.values[:, 1], data.values[:, 5]

# This should not be changes. Later, copy the model to the respective directory 
# if you want to publish it. Check the readme first.
model_file_path = "./model"

model = make_pipeline(StandardScaler(), SVC(gamma="auto", kernel="rbf", verbose=True))
nlp = spacy.load("en_core_web_sm")

token_embeddings, target_ids = preprocessing(features, targets, nlp, label2id)

model = fit_model(model, token_embeddings, target_ids)
store_joblib(model, model_file_path)

