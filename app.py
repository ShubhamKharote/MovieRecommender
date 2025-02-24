import streamlit as st
import pandas as pd
import requests
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit.components.v1 as components

#TMDB API Key
TMDB_API_KEY = "06412c2ac60d3b3a66c7fb129dcaca28&language=en-US"
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# Load dataset
def load_data():
    df = pd.read_csv("tmdb.csv")
    df = df.fillna('')  # Fill NaN values with empty strings
    df['combined_features'] = df['genres'] + ' ' + df['cast'] + ' ' + df['director']
    return df

df = load_data()



def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css() 

# Train TF-IDF model
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

#Sidebar
st.sidebar.title("🎭Movie Menu🎭")
df['genres'] = df['genres'].astype(str).apply(lambda x:re.findall(r"'(.*?)'", str(x)))
genres = sorted(set([genres for sublist in df ['genres'] for genres in sublist ]))
selected_genre = st.sidebar.selectbox("Choose a Genre",["ALL"]+genres)

#fetch movie details from tmdb
def fetch_movie_details(movie_name):
    search_url = f"{TMDB_BASE_URL}/search/movie?api_key={TMDB_API_KEY}&query={movie_name}"
    response = requests.get(search_url).json()

    if response.get("results"):  # Check if results exist
        movie = response["results"][0]  # Get the first search result
        movie_id = movie.get("id", "")
        poster_path = movie.get("poster_path", "")
        overview = movie.get("overview", "Overview not available")

        # Fetch cast details separately
        cast_name = "Cast not available"
        if movie_id:
            credits_url = f"{TMDB_BASE_URL}/movie/{movie_id}/credits?api_key={TMDB_API_KEY}"
            credits_response = requests.get(credits_url).json()
            if "cast" in credits_response and credits_response["cast"]:
                cast_name = ", ".join([cast["name"] for cast in credits_response["cast"][:5]])  # Get top 5 cast members

        homepage_url = f"https://www.themoviedb.org/movie/{movie_id}" if movie_id else "#"
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else ""

        return homepage_url, poster_url, cast_name, overview  # ✅ Return all 4 values

    return "#", "", "Cast not available", "Overview not available"

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
    
    recommendations = []
    for movie in df.iloc[movie_indices]['title'].tolist():
        movie_genres = df.loc[df['title'] == movie, 'genres'].values[0]
        if selected_genre == "All" or selected_genre in df.loc[df['title'] == movie, 'genres'].values[0]:
            homepage_url, poster_url, cast_name, overview = fetch_movie_details(movie)
            recommendations.append({
                "title": movie, 
                "homepage": homepage_url, 
                "poster_url": poster_url, 
                "cast_name": cast_name, 
                "overview": overview
        })           
            
    
    return recommendations

# Streamlit UI
st.title("🎬 Movie Recommender")
st.write("Enter a movie name and get recommendations!")

# User input
movie_name = st.text_input("Enter a movie title:")

if st.button("Get Recommendations"):
    if movie_name:
        recommendations = get_recommendations(movie_name)
        if recommendations:
            st.subheader("Recommended Movies:")
            cols = st.columns(4)
            for idx, movie in enumerate(recommendations):
                with cols[idx % 4]:
                    st.markdown(movie['title'])
                    st.image(movie['poster_url'], width=200)
                    with st.expander(f"More Details"):
                        st.write(f"**Cast:** {movie['cast_name']}")
                        st.write(f"**Overview:** {movie['overview']}")
                        st.markdown(f"[Link]({movie['homepage']})")
        else:
            st.warning("Movie not found or no recommendations available.")
            st.warning("Warning Please select a Genre")    
    else:
        st.warning("Please enter a movie name.")
