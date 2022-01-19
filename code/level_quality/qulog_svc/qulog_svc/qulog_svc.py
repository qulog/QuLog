import os
from typing import List
import spacy
from joblib import load

home = os.path.dirname(os.path.realpath(__file__))

class QulogSVC:
    def __init__(self):
        self.id2label = {0:"info", 1:"error"}

        self.nlp = spacy.load('en_core_web_trf')
        self.model = self.load(os.path.join(home, './model'))


    def load(self, model_file_path):
        return load(model_file_path)


    def _get_trf_embedding(self, doc):
        return doc._.trf_data.tensors[-1].ravel()


    def _get_trf_embeddings(self, trf_docs):
        embeddings = []
        for doc in trf_docs:
            embeddings.append(self._get_trf_embedding(doc))
        return embeddings


    def predict(self, log_line: str):
        return self.predict_batch([log_line])[0]


    def predict_batch(self, log_lines: List[str]):
        docs = list(self.nlp.pipe(log_lines))
        embeddings = self._get_trf_embeddings(docs)
        predictions = self.model.predict(embeddings)
        return [self.id2label[p] for p in predictions]



if __name__ == "__main__":
    q = QulogSVC()
    s = "Hello world"
    p1 = q.predict(s)
    p2 = q.predict_batch([s])
    print("Log level prediction for '{}': {}".format(s, p1))
    print("Log level prediction for '{}': {}".format(s, p2[0]))