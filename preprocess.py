try:
    import librosa
except Exception as e:
    librosa = None
    _librosa_import_error = e
import pandas as pd
import numpy as np
import os
import warnings

# Suppress user warnings from librosa (if available)
warnings.filterwarnings("ignore", category=UserWarning)

def extract_features(file_path, n_mfcc=13, max_pad_len=174):
    """
    Extracts MFCC features from an audio file, pads or truncates them to a fixed length.

    Args:
        file_path (str): Path to the audio file.
        n_mfcc (int): Number of MFCCs to extract.
        max_pad_len (int): The fixed length for padding or truncating.

    Returns:
        np.ndarray: The processed MFCC features, or None if an error occurs.
    """
    if librosa is None:
        raise ImportError(f"librosa is required to extract features but failed to import: {_librosa_import_error}")

    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        
        # Pad or truncate
        if mfccs.shape[1] > max_pad_len:
            mfccs = mfccs[:, :max_pad_len]
        else:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def preprocess_audio_data(data_path, output_csv_path):
    """
    Processes all audio files in a directory, extracts features, and saves them to a CSV.

    Args:
        data_path (str): Path to the directory containing audio files.
        output_csv_path (str): Path to save the output CSV file.
    """
    features_list = []
    
    # Check if data path exists
    if not os.path.exists(data_path):
        print(f"Data path not found: {data_path}")
        return

    # Iterate through all files in the data directory
    for filename in os.listdir(data_path):
        if filename.endswith(('.wav', '.mp3', '.flac', '.ogg')):
            file_path = os.path.join(data_path, filename)
            print(f"Processing {file_path}...")
            
            # Extract features
            features = extract_features(file_path)
            
            if features is not None:
                # Flatten the 2D features array to a 1D array for CSV
                features_flat = features.flatten()
                features_list.append(features_flat)

    if not features_list:
        print("No audio files were processed. Please check the data path and file formats.")
        return

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(features_list)
    df.to_csv(output_csv_path, index=False)
    print(f"Preprocessing complete. Features saved to {output_csv_path}")

if __name__ == '__main__':
    # --- IMPORTANT ---
    # SET THE PATH TO YOUR AUDIO DATASET DIRECTORY HERE
    # For example: 'C:/Users/YourUser/Desktop/audio_clips'
    # Use a raw string or forward slashes to avoid escaping issues on Windows
    PATH_TO_YOUR_AUDIO_DATA = r"D:\\SURYA\\Infosys 6.0\\audio"

    # Set the path for the output CSV file (make absolute in current working dir)
    OUTPUT_CSV = os.path.abspath("processed_audio_features.csv")

    # If librosa failed to import, notify the user and exit early
    if librosa is None:
        print("ERROR: librosa is not available. Install it with: pip install librosa")
        raise SystemExit(1)

    preprocess_audio_data(PATH_TO_YOUR_AUDIO_DATA, OUTPUT_CSV)
