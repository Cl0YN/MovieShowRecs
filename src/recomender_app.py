import streamlit as st
import pandas as pd
from recommender import build_recommender, get_recommendations

net_films = pd.read_csv("net_films.csv")
net_shows = pd.read_csv("net_shows.csv")


cosine_sim_movies, indices_movies = build_recommender(net_films)
cosine_sim_shows, indices_shows = build_recommender(net_shows)


st.title("🎬 Netflix Recommender System")

content_type = st.selectbox("Выберите тип контента:", ["Movie", "TV Show"])

if content_type == "Movie":
    all_titles = net_films['title'].tolist()
    cosine_sim = cosine_sim_movies
    indices = indices_movies
    df = net_films
else:
    all_titles = net_shows['title'].tolist()
    cosine_sim = cosine_sim_shows
    indices = indices_shows
    df = net_shows

title = st.selectbox("Выберите название:", sorted(all_titles))

if st.button("Получить рекомендации"):
    recommendations = get_recommendations(title, cosine_sim, indices, df)
    if isinstance(recommendations, str):
        st.error(recommendations)
    else:
        st.success(f"🎯 Похожие {content_type.lower()}s на: **{title}**")
        for rec in recommendations:
            st.write(f"- {rec}")
