# 🎬 Movie Recommendation System

A content-based movie recommender built with Python and Flask that suggests similar movies based on genres, cast, crew, and keywords from the TMDB dataset.

---

## 🧠 Project Features

- 🔍 Recommend up to 8 similar movies using content-based filtering  
- 🗂 Analyzes genres, cast, director, and keywords for recommendations  
- ✨ Clean and responsive web interface built using HTML + CSS  
- 📊 Uses TF-IDF vectorization to handle text data  
- 🧠 Computes cosine similarity for movie comparison  
- 🔗 Provides direct Google search link for each recommended movie  
- 🚀 Fast, lightweight, and easy to use for learning or demo  

---

## 📁 Dataset

- **Source:** [Kaggle - TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)  
- **Files Used:**
  - `tmdb_5000_movies.csv`
  - `tmdb_5000_credits.csv`
- **Processing:**
  - Merged on movie title  
  - Extracted fields: `genres`, `keywords`, `cast`, `crew`, `overview`  
  - Applied NLP preprocessing (removing spaces, parsing JSON-like strings)  
  - Combined into one column of "tags" used for recommendation  

---

## 🛠 Technologies Used

- 🐍 Python 3.x  
- 📦 Pandas  
- 💡 scikit-learn  
- 🔤 TfidfVectorizer  
- 🌐 Flask  
- 🎨 HTML/CSS (frontend styling)  

---

## ▶️ How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/kavin-S07/Movie-Recommendation-System.git
   cd Movie-Recommendation-System
