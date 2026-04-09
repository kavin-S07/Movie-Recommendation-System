"""
Movie Recommendation System - Model Training
Uses TMDB 5000 dataset + NLP Cosine Similarity
Run this once to train and save the model.
"""

import pandas as pd
import numpy as np
import ast
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import PorterStemmer

# Download nltk data
nltk.download('punkt')

# ── 1. Load Data ─────────────────────────────────────────────────────────────
print("📂 Loading datasets...")
movies_df = pd.read_csv("data/tmdb_5000_movies.csv")
credits_df = pd.read_csv("data/tmdb_5000_credits.csv")

# Merge on title
movies_df = movies_df.merge(credits_df, on="title")
print(f"✅ Loaded {len(movies_df)} movies")

# ── 2. Select Relevant Columns ────────────────────────────────────────────────
movies_df = movies_df[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
movies_df.dropna(inplace=True)

# ── 3. Parse JSON-like Columns ────────────────────────────────────────────────
def extract_names(obj, top_n=None):
    """Extract 'name' field from a JSON-like string list."""
    try:
        items = ast.literal_eval(obj)
        names = [i["name"].replace(" ", "") for i in items]
        return names[:top_n] if top_n else names
    except:
        return []

def extract_director(obj):
    """Extract director name from crew list."""
    try:
        crew = ast.literal_eval(obj)
        for member in crew:
            if member.get("job") == "Director":
                return [member["name"].replace(" ", "")]
        return []
    except:
        return []

print("🔧 Parsing genres, keywords, cast, crew...")
movies_df["genres"]   = movies_df["genres"].apply(extract_names)
movies_df["keywords"] = movies_df["keywords"].apply(extract_names)
movies_df["cast"]     = movies_df["cast"].apply(lambda x: extract_names(x, top_n=3))
movies_df["crew"]     = movies_df["crew"].apply(extract_director)

# ── 4. Process Overview ───────────────────────────────────────────────────────
movies_df["overview"] = movies_df["overview"].apply(lambda x: x.split())

# ── 5. Build Tags Column ──────────────────────────────────────────────────────
movies_df["tags"] = (
    movies_df["overview"] +
    movies_df["genres"] +
    movies_df["keywords"] +
    movies_df["cast"] +
    movies_df["crew"]
)

# Keep only needed columns
final_df = movies_df[["movie_id", "title", "tags"]].copy()
final_df["tags"] = final_df["tags"].apply(lambda x: " ".join(x).lower())

# ── 6. Stemming ───────────────────────────────────────────────────────────────
print("🌿 Applying stemming...")
ps = PorterStemmer()

def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

final_df["tags"] = final_df["tags"].apply(stem)

# ── 7. Vectorize using Bag of Words ──────────────────────────────────────────
print("🔢 Vectorizing tags...")
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(final_df["tags"]).toarray()

# ── 8. Compute Cosine Similarity ─────────────────────────────────────────────
print("📐 Computing cosine similarity matrix...")
similarity = cosine_similarity(vectors)
print(f"✅ Similarity matrix shape: {similarity.shape}")

# ── 9. Save Model ─────────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

with open("model/movies.pkl", "wb") as f:
    pickle.dump(final_df, f)

with open("model/similarity.pkl", "wb") as f:
    pickle.dump(similarity, f)

print("💾 Model saved to model/movies.pkl and model/similarity.pkl")
print("🎉 Training complete! Now run: python app.py")