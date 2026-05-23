# src/tfidf_model.py

import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFRecommender:

    def __init__(self):

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2)
        )

        self.tfidf_matrix = None
        self.df = None

    def fit(self, csv_path):

        self.df = pd.read_csv(csv_path)

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df['clean_text']
        )

        print("TF-IDF model trained.")

    def recommend(self, paper_index, top_k=5):

        similarity = cosine_similarity(
            self.tfidf_matrix[paper_index],
            self.tfidf_matrix
        ).flatten()

        indices = similarity.argsort()[::-1][1:top_k+1]

        recommendations = self.df.iloc[indices][['title']]

        return recommendations

    def save_model(self):

        with open("models/tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)

        with open("models/tfidf_matrix.pkl", "wb") as f:
            pickle.dump(self.tfidf_matrix, f)

        print("TF-IDF model saved.")


if __name__ == "__main__":

    recommender = TFIDFRecommender()

    recommender.fit("Dataset/cleaned_data.csv")

    print(recommender.recommend(0))

    recommender.save_model()