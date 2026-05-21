# src/evaluation.py

import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


def evaluate_similarity(embeddings):

    similarity_matrix = cosine_similarity(embeddings)

    avg_similarity = np.mean(similarity_matrix)

    print(f"Average Similarity Score: {avg_similarity:.4f}")

    return avg_similarity


if __name__ == "__main__":

    embeddings = np.load(
        "models/bert_embeddings.npy"
    )

    evaluate_similarity(embeddings)