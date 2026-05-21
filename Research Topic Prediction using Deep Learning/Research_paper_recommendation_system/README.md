# Research Paper Recommendation System

A Hybrid NLP-Based Research Paper Recommendation System using **TF-IDF**, **BERT Embeddings**, **FAISS**, and **Machine Learning Classification Models** for intelligent and scalable paper recommendations.

---

## Project Overview

This project recommends similar research papers based on semantic meaning, keyword similarity, and topic classification.

The system combines traditional NLP techniques with modern deep learning models to improve recommendation quality and retrieval speed.

The complete pipeline includes:

- Data preprocessing
- TF-IDF based recommendation
- BERT semantic embeddings
- FAISS similarity search
- Topic classification using Random Forest / XGBoost
- Evaluation and comparison of models

---

# Features

## TF-IDF Recommendation System
- Converts research paper text into TF-IDF vectors
- Uses cosine similarity for keyword-based recommendations
- Lightweight and fast baseline model

## BERT Semantic Recommendation
- Uses Sentence Transformers (`all-MiniLM-L6-v2`)
- Generates contextual embeddings
- Improves semantic understanding of papers

## FAISS Similarity Search
- Efficient large-scale vector similarity search
- Fast retrieval for high-dimensional embeddings
- Optimized recommendation performance

## Classification Models
- Random Forest Classifier
- XGBoost Classifier
- Topic/domain prediction for filtered recommendations

## Evaluation Metrics
- Precision@K
- Recall@K
- F1-Score
- Cosine Similarity
- Classification Accuracy

---

# Tech Stack

## Languages & Libraries
- Python
- Pandas
- NumPy
- Scikit-learn
- SentenceTransformers
- FAISS
- XGBoost
- Streamlit

---

# Project Structure

```bash
Research_paper_recommendation_system/
│
├── data/
│   ├── arXiv_dataset.csv
│   └── cleaned_data.csv
│
├── models/
│   ├── bert_embeddings.npy
│   ├── bert_embeddings_5k.npy
│   ├── faiss_index.index
│   ├── tfidf_matrix.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_tfidf_model.ipynb
│   ├── 03_bert_model.ipynb
│   ├── 04_faiss_search.ipynb
│   ├── evaluation.ipynb
│   ├── evaluation_results.csv
│   └── faiss_recommendations.csv
│
├── outputs/
│
├── src/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Dataset

The project uses a research paper dataset containing:
- Paper titles
- Abstracts
- Categories/topics
- Authors

Dataset preprocessing includes:
- Lowercasing
- Stopword removal
- Punctuation cleaning
- Text normalization

---

# Workflow

## 1. Data Preprocessing
- Clean raw text
- Remove null values
- Normalize text
- Save cleaned dataset

Notebook:
```bash
01_preprocessing.ipynb
```

---

## 2. TF-IDF Recommendation Model
- Convert text into TF-IDF vectors
- Compute cosine similarity
- Generate recommendations

Notebook:
```bash
02_tfidf_model.ipynb
```

---

## 3. BERT Embedding Generation
- Generate sentence embeddings
- Capture semantic relationships
- Store embeddings as `.npy` files

Notebook:
```bash
03_bert_model.ipynb
```

---

## 4. FAISS Similarity Search
- Build FAISS index
- Perform fast nearest-neighbor search
- Improve scalability

Notebook:
```bash
04_faiss_search.ipynb
```

---

## 5. Evaluation
- Compare recommendation quality
- Evaluate classification accuracy
- Generate performance tables

Notebook:
```bash
evaluation.ipynb
```

---

# Model Performance (Approximate)

| Model | Performance |
|---|---|
| TF-IDF | ~70–80% relevance |
| BERT | ~85–92% semantic relevance |
| FAISS | ~85–90% retrieval relevance |
| Random Forest | ~80–85% accuracy |
| XGBoost | ~85–90% accuracy |

---

# Installation

## Clone Repository

```bash
git clone https://github.com/parm-08/Research_paper_recommendation_system.git
```

## Navigate to Project

```bash
cd Research_paper_recommendation_system
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Run Jupyter Notebooks

```bash
jupyter notebook
```

Execute notebooks in order:

1. `01_preprocessing.ipynb`
2. `02_tfidf_model.ipynb`
3. `03_bert_model.ipynb`
4. `04_faiss_search.ipynb`
5. `evaluation.ipynb`

---

# Streamlit App

Run the recommendation system interface:

```bash
streamlit run app.py
```

---

# Future Improvements

- Hybrid recommendation engine
- GPU acceleration
- Real-time recommendation API
- Deep learning ranking models
- Research paper summarization
- User-based collaborative filtering

---

# Results

The project successfully demonstrates:

- Semantic recommendation using BERT
- Fast similarity search using FAISS
- Lightweight baseline using TF-IDF
- Topic-aware filtering using ML classifiers
- Detailed evaluation and comparison pipeline

---

# Contribution

Contributions are welcome.

Steps:
1. Fork the repository
2. Create a new branch
3. Commit changes
4. Open a Pull Request

---

# Author

Parmpreet Kaur

---

# License

This project is licensed under the MIT License.