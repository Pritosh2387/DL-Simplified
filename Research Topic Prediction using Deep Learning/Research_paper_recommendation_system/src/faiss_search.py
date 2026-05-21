# src/faiss_search.py

import faiss
import numpy as np
import pandas as pd


class FAISSRecommender:

    def __init__(self):

        self.index = None
        self.df = None

    def build_index(self, embeddings_path, csv_path):

        embeddings = np.load(embeddings_path)

        self.df = pd.read_csv(csv_path)

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embeddings.astype('float32'))

        print("FAISS index created.")

    def recommend(self, query_embedding, top_k=5):

        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            top_k
        )

        recommendations = self.df.iloc[
            indices[0]
        ][['title']]

        return recommendations

    def save_index(self):

        faiss.write_index(
            self.index,
            "models/faiss_index.index"
        )

        print("FAISS index saved.")


if __name__ == "__main__":

    recommender = FAISSRecommender()

    recommender.build_index(
        "models/bert_embeddings.npy",
        "data/cleaned_data.csv"
    )

    recommender.save_index()