import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self, data_path):
        self.data_path = data_path
        self.movie_data = None
        self.feature_vectors = None
        self.similarity = None
        self.vectorizer = TfidfVectorizer()
        self._load_data()
        self._prepare_model()
    
    def _load_data(self):
        """Load and preprocess the movie data"""
        self.movie_data = pd.read_csv(self.data_path)
        
        # Fill null values
        selected_features = ['genres','keywords','tagline','cast','director']
        for feature in selected_features:
            self.movie_data[feature] = self.movie_data[feature].fillna('')
    
    def _prepare_model(self):
        """Prepare the recommendation model"""
        combined_features = self.movie_data['genres'] + ' ' + \
                          self.movie_data['keywords'] + ' ' + \
                          self.movie_data['tagline'] + ' ' + \
                          self.movie_data['cast'] + ' ' + \
                          self.movie_data['director']
        
        self.feature_vectors = self.vectorizer.fit_transform(combined_features)
        self.similarity = cosine_similarity(self.feature_vectors)
    
    def recommend(self, movie_name, count=10):
        """Get movie recommendations based on input movie"""
        list_of_titles = self.movie_data['title'].tolist()
        
        # Find close match for the movie name
        find_close_match = difflib.get_close_matches(movie_name, list_of_titles)
        
        if not find_close_match:
            return []
            
        close_match = find_close_match[0]
        index_of_movie = self.movie_data[self.movie_data.title == close_match]['index'].values[0]
        
        # Get similarity scores
        similarity_score = list(enumerate(self.similarity[index_of_movie]))
        
        # Sort movies based on similarity score
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        
        # Get top recommendations
        recommendations = []
        for i, movie in enumerate(sorted_similar_movies):
            if i == 0:
                continue  # Skip the first one as it's the same movie
            if i > count:
                break
            index = movie[0]
            title = self.movie_data[self.movie_data.index == index]['title'].values[0]
            recommendations.append(title)
            
        return recommendations