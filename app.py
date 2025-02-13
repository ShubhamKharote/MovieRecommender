import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
def load_data():
    df = pd.read_csv("tmdb.csv")
    df = df.fillna('')  # Fill NaN values with empty strings
    df['combined_features'] = df['genres'] + ' ' + df['cast'] + ' ' + df['director']
    return df

df = load_data()

# Train TF-IDF model
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recommendations
def get_recommendations(title):
    title = title.lower().strip()
    idx = df.index[df['title'].str.lower() == title].tolist()
    if not idx:
        return []
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices][['title', 'cast', 'overview', 'homepage']].to_dict(orient='records')

# Streamlit UI
st.title("🎬 Bollywood Movie Recommender")
st.write("Enter a movie name and get recommendations!")

# User input
movie_name = st.text_input("Enter a movie title:")

if st.button("Get Recommendations"):
    if movie_name:
        recommendations = get_recommendations(movie_name)
        if recommendations:
            st.subheader("Recommended Movies:")
            for movie in recommendations:
                with st.expander(movie['title']):
                    st.write(f"**Cast:** {movie['cast']}")
                    st.write(f"**Overview:** {movie['overview']}")
                    st.markdown(f"[More Info]({movie['homepage']})")
        else:
            st.warning("Movie not found or no recommendations available.")
    else:
        st.warning("Please enter a movie name.")
