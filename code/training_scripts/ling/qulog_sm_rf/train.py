import os
import spacy
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from utils import *


#################################################
##### Preprocessing
#################################################


def get_small_web_embeddings(vec):
    embeddings = []
    for doc in vec:
        embeddings.append(doc.vector)
    return embeddings


def preprocessing(features, nlp, buffer=True):
    # Buffering this to prevent recalculation

    docs = get_docs(nlp, features)

    word_class_embeddings_file = "./.trf_word_class_embeddings.pkl"
    if buffer and os.path.isfile(word_class_embeddings_file):
        print("Loading word class embeddings from existing buffer")
        word_class_embeddings = load_pickle(word_class_embeddings_file)
    else:
        word_classses = [" ".join([t.pos_ for t in doc]) for doc in docs]
        word_classs_trf_docs = get_docs(nlp, word_classses)
        word_class_embeddings = get_small_web_embeddings(word_classs_trf_docs)
        if buffer:
            store_pickle(word_class_embeddings, word_class_embeddings_file)
    word_class_embeddings = np.asarray(word_class_embeddings)

    return word_class_embeddings


# Adjust if needed
data = pd.read_csv("./training_data_linguistic_quality.csv")  # here add the linguistic dataset
features, targets = data.values[:, 0], data.values[:, 1]
target_ids = targets.astype("int32")

# This should not be changes. Later, copy the model to the respective directory 
# if you want to publish it. Check the readme first.
model_file_path = "./model"


model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, min_samples_split=2))
nlp = spacy.load('en_core_web_sm')

word_class_embeddings = preprocessing(features, nlp)

#evaluate(model, word_class_embeddings, target_ids, iterations=5)
model = fit_model(model, word_class_embeddings, target_ids)
store_joblib(model, model_file_path)
