# src/preprocessing.py

import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Cleans research paper text.

    Steps:
    - Lowercase conversion
    - Remove special characters
    - Remove stopwords
    - Lemmatization
    """

    text = str(text).lower()

    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    cleaned_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(cleaned_words)


def preprocess_dataset(input_path, output_path):
    """
    Loads dataset and creates cleaned text column.
    """

    df = pd.read_csv(input_path)

    # Combine title + summary
    df['text'] = df['title'] + " " + df['summary']

    # Clean text
    df['clean_text'] = df['text'].apply(clean_text)

    # Save cleaned dataset
    df.to_csv(output_path, index=False)

    print("Dataset preprocessing completed.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":

    preprocess_dataset(
      input_path="data/arXiv_dataset.csv",
      output_path="data/cleaned_data.csv"
    )