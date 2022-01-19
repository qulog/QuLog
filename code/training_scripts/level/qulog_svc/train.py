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


def get_trf_embeddings(trf_docs):
    embeddings = []
    for doc in trf_docs:
        embeddings.append(doc._.trf_data.tensors[-1].ravel())
    return embeddings


def preprocessing(features, targets, nlp, label2id, buffer=True):
    # Buffering this to prevent recalculation
    token_embeddings_file = "./.level_trf_token_embeddings.pkl"
    if buffer and os.path.isfile(token_embeddings_file):
        print("Loading token embeddings from existing buffer")
        token_embeddings = load_pickle(token_embeddings_file)
    else:
        trf_docs = get_docs(nlp, features)
        token_embeddings = get_trf_embeddings(trf_docs)
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
data = pd.read_csv("./training_data_log_level_pred.csv") # here add the log level dataset
features, targets = data.values[:, 0], data.values[:, 1]

# This should not be changes. Later, copy the model to the respective directory 
# if you want to publish it. Check the readme first.
model_file_path = "./model"

label2id = {"info": 0, "error":1}
id2label = {0:"info", 1:"error"}

model = make_pipeline(StandardScaler(), SVC(gamma="auto", kernel="rbf", verbose=True))
nlp = spacy.load('en_core_web_trf')

token_embeddings, target_ids = preprocessing(features, targets, nlp, label2id)

model = fit_model(model, token_embeddings, target_ids)
store_joblib(model, model_file_path)

