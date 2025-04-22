import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    net_films = pd.read_csv('net_films.csv')
    net_shows = pd.read_csv('net_shows.csv')
    return net_films, net_shows

def build_recommender(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    return cosine_sim, indices

def compute_cosine_similarity(df, content_type='Movie'):
    if content_type == 'Movie':
        description = df['description']
    elif content_type == 'TV Show':
        description = df['description']

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(description)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    return cosine_sim, indices

def get_recommendations(title, cosine_sim, indices, df, num_recommendations=10):

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]

    return df['title'].iloc[movie_indices]

