import pandas as pd
import ast
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your datasets
movies_df = pd.read_csv(r"C:\Users\admin\OneDrive\Documents\Kavin CIT\Project\Movie Recommendation system\dataset\tmdb_5000_movies.csv")
credits_df = pd.read_csv(r"C:\Users\admin\OneDrive\Documents\Kavin CIT\Project\Movie Recommendation system\dataset\tmdb_5000_credits.csv")

# Merge on title
movies = movies_df.merge(credits_df, on='title')

# Keep only necessary columns
movies = movies[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']].dropna()

# Convert stringified lists into actual lists
def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

def get_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return i['name']
    return ''

def get_top_cast(obj):
    return [i['name'] for i in ast.literal_eval(obj)[:3]]

# Apply the functions
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(get_top_cast)
movies['crew'] = movies['crew'].apply(get_director)

# Clean text
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: x.replace(" ", "") if isinstance(x, str) else x)

# Save original overview
original_overview = movies['overview'].copy()

# Combine all tags into one string
movies['tags'] = (
        movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew'].apply(lambda x: [x])
)
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = tfidf.fit_transform(movies['tags']).toarray()

# Compute similarity
similarity = cosine_similarity(vectors)

# Final data to save
movie_data = movies[['id', 'title']].copy()
movie_data['overview'] = original_overview
movie_data['tags'] = movies['tags']

# Save
pickle.dump(movie_data, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("✅ Model built and saved as movies.pkl and similarity.pkl")
