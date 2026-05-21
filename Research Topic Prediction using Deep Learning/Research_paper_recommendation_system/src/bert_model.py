# src/bert_model.py

import os
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer


class BERTEmbedder:

    def __init__(self):

        # Lightweight + fast + good quality model
        self.model = SentenceTransformer(
            'all-MiniLM-L6-v2'
        )

    def generate_embeddings(self, csv_path):

        # Load dataset
        df = pd.read_csv(csv_path)

        # -----------------------------------
        # USE SMALLER DATASET FOR DEVELOPMENT
        # Remove this later for final training
        # -----------------------------------
        df = df.head(10000)

        texts = df['clean_text'].astype(str).tolist()

        print(f"Total documents: {len(texts)}")

        # -----------------------------------
        # Generate embeddings efficiently
        # -----------------------------------
        embeddings = self.model.encode(
            texts,

            # Bigger batch = faster
            batch_size=64,

            # Shows progress
            show_progress_bar=True,

            # Direct NumPy output
            convert_to_numpy=True,

            # Faster on CPU
            normalize_embeddings=True
        )

        print("Embeddings generated.")

        return embeddings

    def save_embeddings(self, embeddings):

        os.makedirs("models", exist_ok=True)

        np.save(
            "models/bert_embeddings.npy",
            embeddings
        )

        print("Embeddings saved.")


if __name__ == "__main__":

    bert = BERTEmbedder()

    embeddings = bert.generate_embeddings(
        "data/cleaned_data.csv"
    )

    bert.save_embeddings(embeddings)