from flask import Blueprint, render_template, request, jsonify
from .recommender import MovieRecommender
import os

main = Blueprint('main', __name__)

# Initialize recommender
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'movies.csv')
recommender = MovieRecommender(data_path)

@main.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        movie_name = request.form['movie_name']
        recommendations = recommender.recommend(movie_name)
        return jsonify(recommendations)
    return render_template('index.html')

@main.route('/recommend', methods=['POST'])
def get_recommendations():
    movie_name = request.json.get('movie_name', '')
    recommendations = recommender.recommend(movie_name)
    return jsonify(recommendations)