import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
import librosa
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained model
model = joblib.load('PPDMD5_SS.pkl')
scaler = joblib.load('scaler.joblib')

# Adding exception class
class AppError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def extract_features_from_audio(file):
    # Load the audio file
    y, sr = librosa.load(file, sr=None)

    # Extract features from the audio file
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

def get_extract_features(mfccs_mean):
    mfcc_features = mfccs_mean
    if mfcc_features is not None and mfcc_features.size > 0:
        mfcc_features = mfcc_features.reshape(1, -1)  # Reshape for the scaler
        scaled_features = scaler.transform(mfcc_features)
        return scaled_features
    else:
        raise AppError("Error: Extracted features are empty")

def classify_music(features):
    # Get decision scores for each class
    decision_scores = model.decision_function(features)[0]
    
    # Convert decision scores to probabilities (softmax)
    probabilities = np.exp(decision_scores) / np.sum(np.exp(decision_scores))
    
    # Define genres (make sure these match your model's training classes)
    genres = ["blues", "classical", "pop", "reggae", "rock"]
    
    return dict(zip(genres, probabilities))

def get_similar_songs(genre, input_features):
    file_path = f'songs/scaled_{genre}.csv'
    if os.path.exists(file_path):
        genre_data = pd.read_csv(file_path)
        genre_features = genre_data.iloc[:, :-2].values  # All columns except the last two (filename and label)
        cosine_similarities = cosine_similarity(input_features, genre_features)[0]

 # Print the cosine similarity results
        print("Cosine Similarities:", cosine_similarities)

        top_indices = np.argsort(cosine_similarities)[-10:][::-1]
        top_songs = genre_data.iloc[top_indices]
        return top_songs['filename'].tolist()
    else:
        return []

def main():
    st.title("Sistem Rekomendasi Musik")
    st.subheader("Kelompok D5")
    st.caption("Anggota Kelompok: \n"
               "1. I Wayan Gede Gemuh Raharja RL (2208561004)\n"
               "2. Ryan Hangralim (2208561030)\n"
               "3. Komang Gede Bagus Devit Aditiya (2208561073)\n"
               "4. Putu Ananda Darma Wiguna (2208561099)")
    st.divider()

    st.subheader("Upload Wav File")
    upload_file = st.file_uploader("Upload file", type=["wav"])

    if upload_file is not None:
        st.write("File uploaded")
        st.audio(upload_file)
        
        try:
            # Extract and scale features from the uploaded audio file
            mfccs_mean = extract_features_from_audio(upload_file)
            features = get_extract_features(mfccs_mean)
            results = classify_music(features)

            st.subheader("Classification Result")
            for genre, probability in results.items():
                st.write(f"{genre}: {probability*100:.2f}%")

            # Plot the results as a horizontal bar chart
            fig, ax = plt.subplots()
            genres = list(results.keys())
            probabilities = list(results.values())
            ax.barh(genres, probabilities, color='skyblue')
            ax.set_xlabel('Probability')
            ax.set_title('Music Genre Classification Result')
            
            st.pyplot(fig)

            # Get the most likely genre
            most_likely_genre = max(results, key=results.get)
            st.subheader(f"Music Recommendation")
            similar_songs = get_similar_songs(most_likely_genre, features)

            for i, song in enumerate(similar_songs, 1):
                st.write(f"{i}. {song}")

                file_path = f'songs_file/{most_likely_genre}/{song}'
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        audio_bytes = f.read()
                    st.audio(audio_bytes, format='audio/wav')
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
