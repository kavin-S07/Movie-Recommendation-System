"""
Movie Recommendation System - Flask Backend
Downloads model files from Google Drive on first startup.
"""

from flask import Flask, request, jsonify, render_template
import pickle
import os
import gdown

app = Flask(__name__)

# ── Direct Google Drive URLs (UPDATED) ────────────────────────────────────────
MOVIES_URL = "https://drive.google.com/uc?id=1KfBHxq-Nz-d6r9APS5hYEQude4lpHkpJ"
SIMILARITY_URL = "https://drive.google.com/uc?id=1ZI_jEEEWsv-kZpETDdnYnUZxkWZk5DB2"

MOVIES_PATH = "model/movies.pkl"
SIMILARITY_PATH = "model/similarity.pkl"

# ── Download using gdown ──────────────────────────────────────────────────────
def download_file(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    print(f"⬇️ Downloading {dest_path}...")
    gdown.download(url, dest_path, quiet=False)

    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"✅ Saved {dest_path} ({size_mb:.1f} MB)")


# ── Load or Download Model ────────────────────────────────────────────────────
def load_model():
    global movies, similarity

    # 🚨 Remove corrupted small files
    if os.path.exists(MOVIES_PATH) and os.path.getsize(MOVIES_PATH) < 1_000_000:
        print("⚠ Removing corrupted movies.pkl")
        os.remove(MOVIES_PATH)

    if os.path.exists(SIMILARITY_PATH) and os.path.getsize(SIMILARITY_PATH) < 10_000_000:
        print("⚠ Removing corrupted similarity.pkl")
        os.remove(SIMILARITY_PATH)

    # Download if not exists
    if not os.path.exists(MOVIES_PATH):
        download_file(MOVIES_URL, MOVIES_PATH)
    else:
        print(f"✅ Found cached {MOVIES_PATH}")

    if not os.path.exists(SIMILARITY_PATH):
        download_file(SIMILARITY_URL, SIMILARITY_PATH)
    else:
        print(f"✅ Found cached {SIMILARITY_PATH}")

    print("📦 Loading model into memory...")

    with open(MOVIES_PATH, "rb") as f:
        movies = pickle.load(f)

    with open(SIMILARITY_PATH, "rb") as f:
        similarity = pickle.load(f)

    print(f"🎬 Model ready — {len(movies)} movies available")


# Load model at startup
load_model()


# ── Recommendation Logic ──────────────────────────────────────────────────────
def recommend(movie_title, top_n=8):
    movie_title = movie_title.strip().lower()

    matched = movies[movies["title"].str.lower() == movie_title]

    if matched.empty:
        matched = movies[movies["title"].str.lower().str.contains(movie_title, na=False)]

    if matched.empty:
        return []

    movie_index = matched.index[0]
    distances = similarity[movie_index]

    similar_indices = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:top_n + 1]

    return [
        {
            "title": movies.iloc[idx]["title"],
            "score": round(float(score) * 100, 1)
        }
        for idx, score in similar_indices
    ]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def get_recommendations():
    data = request.get_json()
    movie_name = data.get("movie", "").strip()

    if not movie_name:
        return jsonify({"error": "Please enter a movie name"}), 400

    recommendations = recommend(movie_name)

    if not recommendations:
        return jsonify({
            "error": f'Movie "{movie_name}" not found. Try another title.'
        }), 404

    return jsonify({"movie": movie_name, "recommendations": recommendations})


@app.route("/search", methods=["GET"])
def search_movies():
    query = request.args.get("q", "").strip().lower()

    if len(query) < 2:
        return jsonify([])

    matches = movies[movies["title"].str.lower().str.contains(query, na=False)]
    titles = matches["title"].head(8).tolist()

    return jsonify(titles)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "movies": len(movies)}), 200


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)