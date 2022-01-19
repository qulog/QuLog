
import pickle
import numpy as np
from joblib import dump

from sklearn.metrics import f1_score


#################################################
##### Storage
#################################################

def store_pickle(object, file_path):
    with open(file_path, 'wb') as fh:
        pickle.dump(object, fh)


def load_pickle(file_path):
    with open(file_path, 'rb') as fh:
        object = pickle.load(fh)
    return object


def store_joblib(model, model_file_path):
    print("Storing model as {}...".format(model_file_path))
    dump(model, model_file_path)
    print("Model storing done.")


#################################################
##### Preprocessing
#################################################

def get_docs(nlp, log_messages):    
    print("Calculating embeddings...")
    docs = list(nlp.pipe(log_messages))
    print("Embedding calculation done.")
    return docs


#################################################
##### Evaluation
################################################# 

def fit_model(model, features, targets):
    print("-------" * 10)
    print("Training log level quality checking model...")

    model.fit(features, targets)

    print("Training done.")
    print("-------" * 10)

    return model


def evaluate(model, token_embeddings, target_ids, iterations=10, split=0.7):
    """Runs evaluation on a model for a number of iteration using a defined split """

    f1_scores = []
    sample_size = len(target_ids)
    for i in range(iterations):
        print("-------" * 10)
        print("Evaluating phase {} / 10".format(i+1))
        train_indecies = np.random.randint(0, sample_size, size=int(split * sample_size))
        test_indecies = list(set(np.arange(sample_size)).difference(set(train_indecies)))
        train_x, train_y = token_embeddings[train_indecies, :], target_ids[train_indecies]

        fit_model(model, train_x, train_y)

        print("-------" * 10)

        test_x, test_y = token_embeddings[test_indecies], target_ids[test_indecies]
        pred_y = model.predict(test_x)
        f1_scores.append(f1_score(pred_y, test_y))

        print("The F1 score is {}".format(f1_scores[i]))
        print("-------" * 10)

    return f1_scores

