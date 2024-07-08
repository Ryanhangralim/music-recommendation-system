import pandas as pd
import joblib
import os

# Load the dataset
file_path = 'Ekstraksi_MFCC.csv'
mfcc_data = pd.read_csv(file_path)

# Load the scaler
scaler_path = 'scaler.joblib'
scaler = joblib.load(scaler_path)

# Apply the scaler to the MFCC features
mfcc_features = mfcc_data.iloc[:, 1:14]
scaled_features = scaler.transform(mfcc_features)

# Create a new DataFrame with scaled features and the original labels
scaled_mfcc_data = pd.DataFrame(scaled_features, columns=mfcc_data.columns[1:14])
scaled_mfcc_data['filename'] = mfcc_data['filename'].apply(lambda x: os.path.basename(x))
scaled_mfcc_data['label'] = mfcc_data['label']

# Function to save scaled data by genre
def save_genre_specific_csv(df, genre):
    genre_df = df[df['label'] == genre]
    genre_df.to_csv(f'songs/scaled_{genre}.csv', index=False)

# Get unique genres
genres = mfcc_data['label'].unique()

# Save each genre-specific dataset into a different CSV file
for genre in genres:
    save_genre_specific_csv(scaled_mfcc_data, genre)
