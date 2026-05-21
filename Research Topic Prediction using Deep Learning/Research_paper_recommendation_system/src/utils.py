# src/utils.py

import pandas as pd


def load_dataset(path):

    return pd.read_csv(path)


def display_recommendations(recommendations):

    for idx, row in recommendations.iterrows():

        print(f"{idx+1}. {row['title']}")