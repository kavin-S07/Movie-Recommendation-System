🎬 Movie Recommendation System
A content-based movie recommendation system built using machine learning and deployed with a Flask web interface.

📌 Project Overview
This system recommends movies similar to the one the user enters. It uses content-based filtering by analyzing genres, cast, crew, and keywords from the movie metadata. The system uses TF-IDF vectorization and cosine similarity to find the most relevant movies.

📁 Project Structure
pgsql
Copy
Edit
Movie-Recommendation-System/
├── dataset/
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
├── templates/
│   └── index.html
├── build_model.py
├── app.py
├── movies.pkl
├── similarity.pkl
└── README.md
⚙️ How It Works
Step 1: Build the Model

Run the script below to clean the data, extract features, and compute similarity between movies.

nginx
Copy
Edit
python build_model.py
This will generate two files:

movies.pkl: Cleaned movie data

similarity.pkl: Cosine similarity matrix

Step 2: Run the Web App

Launch the Flask app:

nginx
Copy
Edit
python app.py
Then open your browser and go to:

cpp
Copy
Edit
http://127.0.0.1:5000/
✨ Features
Suggests 8 similar movies for any input movie

Dark-themed user interface

Shows:

Movie Title

Tagline

IMDB Rating

Runtime

Genres

Google search link

🛠 Technologies Used
Python

Pandas

Scikit-learn

Flask

HTML/CSS

📦 Dataset Source
TMDB 5000 Movie Dataset from Kaggle
Link: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

✅ Sample Output
Input: Iron Man

Recommendations:

Iron Man 2

Iron Man 3

Avengers: Age of Ultron

Ant-Man

Captain America: Civil War

The Avengers

Thor

Guardians of the Galaxy
