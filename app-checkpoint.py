import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Movie Recommender (No API Version)")

# ---- DATA ----
data = {
    'title': [
        'Inception', 'Interstellar', 'The Matrix',
        'Avengers', 'Iron Man', 'Titanic',
        'The Dark Knight', 'Avatar', 'Joker', 'Tenet'
    ],

    'overview': [
        'dreams within dreams mind bending thriller',
        'space time travel black holes nasa mission',
        'simulation reality hacker dystopian world',
        'superhero team saves world marvel action',
        'genius builds iron suit technology hero',
        'romantic tragedy ship sinks love story',
        'batman fights joker crime dark city',
        'alien world exploration sci fi adventure',
        'mental illness transformation dark character',
        'time inversion spy mission sci fi thriller'
    ],

    'genre': [
        'sci-fi thriller', 'sci-fi space', 'sci-fi action',
        'action superhero', 'action tech', 'romance drama',
        'action crime', 'sci-fi fantasy', 'drama crime', 'sci-fi action'
    ]
}

df = pd.DataFrame(data)

df['combined'] = df['overview'] + " " + df['genre']

# ---- NLP ----
vectorizer = TfidfVectorizer(stop_words='english')
matrix = vectorizer.fit_transform(df['combined'])
similarity = cosine_similarity(matrix)

# ---- FUNCTION ----
def recommend(movie):
    idx = df[df['title'] == movie].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores[1:6]

# ---- UI ----
movie = st.selectbox("Pick a movie", df['title'])

if st.button("Recommend"):
    results = recommend(movie)

    st.subheader("Top Recommendations:")

    for i in results:
        st.write( df.iloc[i[0]]['title'])
        st.write("Genre:", df.iloc[i[0]]['genre'])
        st.write("---")