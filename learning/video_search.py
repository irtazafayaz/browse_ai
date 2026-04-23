import streamlit as st
import pickle
import torch
from sentence_transformers import util

# Load the saved DataFrame with embeddings
with open('test/dataset/saved_df.pkl', 'rb') as f:
    df = pickle.load(f)


# Define sorted recommendation function based on precomputed embeddings
def get_sorted_recommendations(video_id, top_n=5):
    selected_embedding = df[df['Video ID'] == video_id]['embedding'].values[0]

    # Compute cosine similarities
    similarities = [(i, util.cos_sim(selected_embedding, emb).item()) for i, emb in enumerate(df['embedding'])]

    # Sort similarities from highest to lowest, excluding the selected video itself
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    recommended_indices = [index for index, score in sorted_similarities[1:top_n + 1]]

    # Create sorted recommendation DataFrame
    recommendations = df.iloc[recommended_indices]
    recommendations = recommendations[
        ['Video ID', 'Title', 'Chapter Number', 'Duration', 'Description', 'URL', 'Rating', 'Level', 'Language']]
    recommendations['Similarity Rank'] = range(1, len(recommendations) + 1)

    return recommendations


# Streamlit App Layout
st.title("Chapter-wise Video Recommendation System")
st.subheader("Find videos chapter-wise, similar to YouTube's recommendation system.")

# Sidebar Input for Video ID
video_id_input = st.sidebar.selectbox("Select a Video ID", df['Video ID'].unique())
st.write(f"### Now Playing: {video_id_input}")

# Dummy Video Player (Placeholder for actual video content)
st.video("https://dummyvideo.com")  # Replace with actual video URL if available

# Display Recommended Videos in the Sidebar
st.sidebar.header("Top Recommendations")
recommendations = get_sorted_recommendations(video_id_input, top_n=5)

# Display the list of recommendations
for idx, row in recommendations.iterrows():
    if st.sidebar.button(f"{row['Similarity Rank']}. {row['Title']}"):
        video_id_input = row['Video ID']  # Update the main display with selected recommendation

# Display Recommendations on the Right Side of the Main Section
st.write("### Recommended Videos")
for idx, row in recommendations.iterrows():
    st.write(f"**{row['Title']}**")
    st.write(
        f"Chapter: {row['Chapter Number']} | Duration: {row['Duration']} | Rating: {row['Rating']} | Level: {row['Level']} | Language: {row['Language']}")
    st.write(f"Description: {row['Description']}")
    st.write(f"[Watch Video]({row['URL']})")
    st.write("---")