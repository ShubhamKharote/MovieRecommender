import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import streamlit.components.v1 as components

# TMDb API Key (Replace with your own key)
TMDB_API_KEY = "06412c2ac60d3b3a66c7fb129dcaca28&language=en-US"
TMDB_BASE_URL = "https://api.themoviedb.org/3"

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

# Sidebar for genre selection
st.sidebar.title("🎭 Select Movie Genre")

df['genres'] = df['genres'].astype(str).apply(lambda x: re.findall(r"'(.*?)'", x))
genres = sorted(set(genre for sublist in df['genres'] for genre in sublist))
selected_genre = st.sidebar.selectbox("Choose a genre", ["All"] + genres)

# Fetch available years from TMDb
def fetch_available_years():
    years = []
    for year in range(1900, 2025):  # Checking from 1900 to the current year
        search_url = f"{TMDB_BASE_URL}/discover/movie?api_key={TMDB_API_KEY}&primary_release_year={year}"
        response = requests.get(search_url).json()
        if response.get("total_results", 0) > 0:
            years.append(year)
    return sorted(years, reverse=True)

# Sidebar for year selection
st.sidebar.title("📅 Select Release Year")
available_years = fetch_available_years()
selected_year = st.sidebar.selectbox("Choose a year", ["All"] + available_years)

# Function to fetch movie details from TMDb
def fetch_movie_details(movie_name):
    search_url = f"{TMDB_BASE_URL}/search/movie?api_key={TMDB_API_KEY}&query={movie_name}"
    response = requests.get(search_url).json()
    if response.get("results"):  # Check if results exist
        movie = response["results"][0]  # Get the first search result
        movie_id = movie.get("id", "")
        poster_path = movie.get("poster_path", "")
        overview = movie.get("overview", "Overview not available")
        release_year = movie.get("release_date", "Unknown")[:4]

        # Fetch cast details separately
        cast_name = "Cast not available"
        if movie_id:
            credits_url = f"{TMDB_BASE_URL}/movie/{movie_id}/credits?api_key={TMDB_API_KEY}"
            credits_response = requests.get(credits_url).json()
            if "cast" in credits_response and credits_response["cast"]:
                cast_name = ", ".join([cast["name"] for cast in credits_response["cast"][:5]])  # Get top 5 cast members

        homepage_url = f"https://www.themoviedb.org/movie/{movie_id}" if movie_id else "#"
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else ""

        return homepage_url, poster_url, cast_name, overview, release_year  # ✅ Return all 5 values

    return "#", "", "Cast not available", "Overview not available", "Unknown"

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
        homepage_url, poster_url, cast_name, overview, release_year = fetch_movie_details(movie)
        if (selected_genre == "All" or selected_genre in movie_genres) and (selected_year == "All" or selected_year == release_year):
            recommendations.append({
                "title": movie, 
                "homepage": homepage_url, 
                "poster_url": poster_url, 
                "cast_name": cast_name, 
                "overview": overview,
                "release_year": release_year
            })
    return recommendations

# Streamlit UI
st.title("🎬 Bollywood Movie Recommender")
st.write("Enter a movie name and get recommendations with posters and IMDb links!")

# User input
movie_name = st.text_input("Enter a movie title:")

if st.button("Get Recommendations"):
    if movie_name:
        recommendations = get_recommendations(movie_name)
        if recommendations:
            st.subheader("Recommended Movies:")
            cols = st.columns(2)  # Display in a 2-column grid
            for idx, movie in enumerate(recommendations):
                with cols[idx % 2]:  # Alternating between columns
                    st.image(movie['poster_url'], width=200)
                    st.markdown(f"### {movie['title']} ({movie['release_year']})")
                    with st.expander("More Details"):
                        st.write(f"**Cast:** {movie['cast_name']}")
                        st.write(f"**Overview:** {movie['overview']}")
                        st.markdown(f"[More Info]({movie['homepage']})")
        else:
            st.warning("Movie not found or no recommendations available.")
    else:
        st.warning("Please enter a movie name.")
