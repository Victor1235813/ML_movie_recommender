from sklearn.neighbors import NearestNeighbors
from thefuzz import fuzz
import numpy as np

class Recommender:
    def __init__(self, metric, algorithm, k, data, decode_id_movie):
        self.metric = metric
        self.algorithm = algorithm
        self.k = k
        self.data = data
        self.decode_id_movie = decode_id_movie
        self.model = self._recommender().fit(data)
    
    def make_recommendation(self, new_movie, n_recommendations):
        recommended = self._recommend(new_movie=new_movie, n_recommendations=n_recommendations)
        print("... Done")
        return recommended 
    
    def _recommender(self):
        return NearestNeighbors(metric=self.metric, algorithm=self.algorithm, n_neighbors=self.k, n_jobs=-1)
    
    def _recommend(self, new_movie, n_recommendations):
        # Get the id of the recommended movies
        recommendations = []
        recommendation_ids = self._get_recommendations(new_movie=new_movie, n_recommendations=n_recommendations)
        # return the name of the movie using a mapping dictionary
        recommendations_map = self._map_indices_to_movie_title(recommendation_ids)
        # Translate this recommendations into the ranking of movie titles recommended
        for i, (idx, dist) in enumerate(recommendation_ids):
            recommendations.append(recommendations_map[idx])
        return recommendations
                 
    def _get_recommendations(self, new_movie, n_recommendations):
        # Get the id of the song according to the text
        recom_movie_id = self._fuzzy_matching(movie=new_movie)
        # Start the recommendation process
        print(f"Starting the recommendation process for {new_movie} with id of {recom_movie_id}...")
        # Return the n neighbors for the song id
        distances, indices = self.model.kneighbors(self.data[recom_movie_id], n_neighbors=n_recommendations+1)
        return sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    
    def _map_indices_to_movie_title(self, recommendation_ids):
        # get reverse mapper
        return {movie_id: movie_title for movie_title, movie_id in self.decode_id_movie.items()}
    
    def _fuzzy_matching(self, movie):
        match_tuple = []
        # get match
        for title, idx in self.decode_id_movie.items():
            ratio = fuzz.ratio(title.lower(), movie.lower())
            if ratio >= 60:
                match_tuple.append((title, idx, ratio))
        # sort
        match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
        if not match_tuple:
            print(f"The recommendation system could not find a match for {movie}")
            return
        return match_tuple[0][1]