from flask import Flask, request, render_template
import pickle
import pandas as pd
import ast

app = Flask(__name__)

# Load model
data = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Load full movie info for additional details
full_movies_df = pd.read_csv(r"C:\Users\admin\OneDrive\Documents\Kavin CIT\Project\Movie Recommendation system\dataset\tmdb_5000_movies.csv")
full_movies_df.set_index('title', inplace=True)

def get_movie_details(title):
    try:
        row = full_movies_df.loc[title]
        genres_raw = ast.literal_eval(row.get('genres', '[]'))
        genres = genres_raw[0]['name'] if genres_raw else 'N/A'
        return {
            'tagline': row.get('tagline', 'N/A'),
            'vote_average': row.get('vote_average', 'N/A'),
            'runtime': row.get('runtime', 'N/A'),
            'genres': genres
        }
    except:
        return {
            'tagline': 'N/A',
            'vote_average': 'N/A',
            'runtime': 'N/A',
            'genres': 'N/A'
        }

def recommend(movie_name):
    movie_name = movie_name.lower()
    movie_index = data[data['title'].str.lower() == movie_name].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:9]  # Top 8

    recommendations = []
    for i in movie_list:
        row = data.iloc[i[0]]
        title = row.title
        details = get_movie_details(title)
        google_link = f"https://www.google.com/search?q={'+'.join(title.split())}+movie"
        recommendations.append({
            'title': title,
            'tagline': details['tagline'],
            'vote_average': details['vote_average'],
            'runtime': details['runtime'],
            'genres': details['genres'],
            'link': google_link
        })
    return recommendations

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        movie_name = request.form['movie']
        try:
            recommendations = recommend(movie_name)
        except:
            recommendations = []
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
