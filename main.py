import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
import librosa
import os
import random

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

def get_similar_songs(genre):
    folder_path = f"Dataset/{genre}"
    if os.path.exists(folder_path):
        all_songs = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        random.shuffle(all_songs)
        return all_songs[:10]
    else:
        return []

def main():
    st.title("Klasifikasi Genre Musik")
    st.subheader("Kelompok D5")
    st.caption("Anggota Kelompok: \n"
               "1. I Wayan Gede Gemuh Raharja RL (2208561004)\n"
               "2. Ryan Hangralim (2208561030)\n"
               "3. Komang Gede Bagus Devit Aditiya (2208561073)\n"
               "4.  Putu Ananda Darma Wiguna (2208561099)")
    st.divider()

    st.subheader("Unggah File WAV")
    upload_file = st.file_uploader("Unggah file", type=["wav"])

    if upload_file is not None:
        st.write("Anda telah berhasil mengunggah file")
        
        # Simulate music classification
        try:
            mfccs_mean = extract_features_from_audio(upload_file)
            features = get_extract_features(mfccs_mean)
            results = classify_music(features)

            st.subheader("Hasil Klasifikasi")
            for genre, probability in results.items():
                st.write(f"{genre}: {probability*100:.2f}%")

            # Plot the results as a horizontal bar chart
            fig, ax = plt.subplots()
            genres = list(results.keys())
            probabilities = list(results.values())
            ax.barh(genres, probabilities, color='skyblue')
            ax.set_xlabel('Probabilitas')
            ax.set_title('Hasil Klasifikasi Genre Musik')
            
            st.pyplot(fig)

            # Get the most likely genre
            most_likely_genre = max(results, key=results.get)
            st.subheader(f"Lagu-Lagu Mirip dalam Genre {most_likely_genre}")
            similar_songs = get_similar_songs(most_likely_genre)

            for i, song in enumerate(similar_songs, 1):
                st.write(f"{i}. {song}")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()