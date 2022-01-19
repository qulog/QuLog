import os
import spacy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
        trf_docs = get_docs(nlp, features)
        # token_embeddings = get_trf_embeddings(trf_docs)
        token_embeddings = get_small_web_embeddings(trf_docs)
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


# Adjust if needed
data = pd.read_csv("./filtered_log_df.csv")   # here add the log level dataset
label2id = {"info": 0, "debug": 0, "trace": 0, "warning": 1, "error":1, "exception": 1, "critical": 1}

data["log_level"] = data["log_level"].apply(lambda x: "warning" if x == "warn" else x)
data = data[data['log_level'].isin(label2id)]
print("Dataset size: {}".format(len(data)))
features, targets = data.values[:, 1], data.values[:, 5]


# This should not be changes. Later, copy the model to the respective directory 
# if you want to publish it. Check the readme first.
model_file_path = "../../../level_quality/level_qulog_sm_rf/level_qulog_sm_rf/model"

# label2id = {"info": 0, "error":1}
id2label = {0:"info", 1:"error"}

model = make_pipeline(StandardScaler(), RandomForestClassifier())
nlp = spacy.load("en_core_web_sm")

token_embeddings, target_ids = preprocessing(features, targets, nlp, label2id)

model = fit_model(model, token_embeddings, target_ids)
store_joblib(model, model_file_path)

