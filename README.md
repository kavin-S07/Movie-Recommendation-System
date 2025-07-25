[🎬 Movie Recommendation System
This is a content-based movie recommendation system built using Python, Flask, and the TMDB 5000 Movie Dataset. The system suggests similar movies based on user input by analyzing genres, keywords, cast, and crew using TF-IDF and cosine similarity.

📂 Project Structure
bash
Copy
Edit
Movie-Recommendation-System/
│
├── build_model.py           # Builds and saves the model (movies.pkl, similarity.pkl)
├── app.py                   # Flask web application
├── movies.pkl               # Pickled movie data with tags
├── similarity.pkl           # Pickled cosine similarity matrix
├── requirements.txt         # Python dependencies
├── templates/
│   └── index.html           # Web page template (HTML with Jinja2)
└── README.md                # Project documentation
📊 Dataset Used
TMDB 5000 Movie Dataset

Files used:

tmdb_5000_movies.csv

tmdb_5000_credits.csv

⚙️ How It Works
build_model.py:

Cleans and preprocesses movie data.

Extracts important features like genres, keywords, cast, and director.

Uses TF-IDF vectorization and cosine similarity to compute movie similarities.

Saves data using Pickle.

app.py:

Loads the saved data and model.

Uses Flask to create a web interface.

Returns top 8 recommended movies with details like title, tagline, rating, runtime, and genre.

index.html:

A clean and dark-themed UI where users enter a movie name and get recommendations in styled cards.
](http://127.0.0.1:5000/
)
