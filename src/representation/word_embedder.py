import gensim.downloader as api
import numpy as np

class WordEmbedder:
    """
    Word embedding class using pre-trained models from gensim.
    """

    def __init__(self, model_name="glove-wiki-gigaword-50"):
        print(f"🔹 Loading model: {model_name} ...")
        self.model = api.load(model_name)
        print("✅ Model loaded successfully.")
        self.vector_size = self.model.vector_size

    def get_vector(self, word):
        try:
            return self.model[word]
        except KeyError:
            return np.zeros(self.vector_size)

    def get_similarity(self, w1, w2):
        if w1 not in self.model.key_to_index or w2 not in self.model.key_to_index:
            return 0.0
        return self.model.similarity(w1, w2)

    def get_most_similar(self, word, top_n=10):
        if word not in self.model.key_to_index:
            return []
        return self.model.most_similar(word, topn=top_n)

    def embed_document(self, document):
        tokens = document.lower().split()
        vectors = [self.get_vector(t) for t in tokens if t in self.model.key_to_index]
        if not vectors:
            return np.zeros(self.vector_size)
        return np.mean(vectors, axis=0)
