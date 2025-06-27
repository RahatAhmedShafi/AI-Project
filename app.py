import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import requests

def fetch_poster(title, api_key):
    url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("Response") == "True" and data.get("Poster") and data.get("Poster") != "N/A":
            return data["Poster"]
    except Exception:
        pass
    return None

OMDB_API_KEY = "57583f37"
# Load data
@st.cache_data
def load_data():
    path = "d:/AI Project/Movie_Database.csv"
    df = pd.read_csv(path, encoding='latin1')
    df['Genres'] = df['Genres'].apply(
        lambda x: [genre.strip().lower() for genre in x.split(',')] if isinstance(x, str) else []
    )
    df['combined_features'] = df['Movie'] + ' ' + \
                              df['Writer'] + ' ' + \
                              df['Director'] + ' ' + \
                              df['Genres'].apply(lambda x: ' '.join(x)) + ' ' + \
                              df['Certification/Rating']
    df['combined_features'] = df['combined_features'].fillna('')
    return df

df = load_data()

# TF-IDF and similarity
@st.cache_resource
def get_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = get_similarity(df)
title_indices = pd.Series(df.index, index=df['Movie'].str.lower())

def get_recommendations(title, df, cosine_sim):
    title = title.lower()
    if title not in title_indices:
        return []
    idx = title_indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices][['Movie', 'Year', 'Genres', 'Director']]

# Streamlit UI
st.title("🎬 Movie Recommendation System")

movie_list = df['Movie'].sort_values().tolist()
search_title = st.selectbox(
    "Type or select a movie title:",
    movie_list
)

if st.button("Recommend"):
    if not search_title or search_title.strip() == "":
        st.warning("Please enter or select a movie title.")
    else:
        recs = get_recommendations(search_title, df, cosine_sim)
        if isinstance(recs, list) and not recs:
            st.warning("No recommendations found.")
        else:
            st.subheader("Top 5 Recommendations:")
            cols = st.columns(len(recs))
            for idx, (i, row) in enumerate(recs.iterrows()):
                with cols[idx]:
                    poster_url = fetch_poster(row['Movie'], OMDB_API_KEY)
                    if poster_url:
                        st.image(poster_url, width=150)
                    else:
                        st.info("No poster found.")
                    st.markdown(f"**{row['Movie']}** ({row['Year']})")
                    st.markdown(f"Genres: {', '.join(row['Genres'])}")
                    st.markdown(f"Director: {row['Director']}")