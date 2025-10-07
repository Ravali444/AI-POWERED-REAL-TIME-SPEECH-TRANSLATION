import os
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
import numpy as np
merged_audio_folder = r"C:\Users\sksin\Desktop\machine learning\Speech_Translation\merged_dataset\audio"
preprocessed_folder = os.path.join(os.path.dirname(merged_audio_folder), "preprocessed_audio")
os.makedirs(preprocessed_folder, exist_ok=True)

metadata_csv = os.path.join(os.path.dirname(merged_audio_folder), "metadata.csv")
df = pd.read_csv(metadata_csv)

new_audio_files = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    old_file = os.path.join(merged_audio_folder, row['audio_file'])
    
    if not os.path.exists(old_file):
        print(f"⚠️ Missing file: {row['audio_file']}")
        continue
    
    # Load and resample to 16 kHz
    y, sr = librosa.load(old_file, sr=16000, mono=True)
    
    # Normalize
    y = y / max(abs(y))
    
    # Create prefixed filename
    if row['language'] == 'en':
        new_file_name = f"en_{row['audio_file']}"
    elif row['language'] == 'hi':
        new_file_name = f"hi_{row['audio_file']}"
    else:
        new_file_name = row['audio_file']
    
    out_path = os.path.join(preprocessed_folder, new_file_name)
    
    # Save preprocessed audio
    sf.write(out_path, y, 16000)
    
    # Update metadata
    new_audio_files.append([new_file_name, row['text'], row['language']])
  # Save updated metadata
df_preprocessed = pd.DataFrame(new_audio_files, columns=["audio_file", "text", "language"])
preprocessed_csv = os.path.join(os.path.dirname(preprocessed_folder), "metadata_preprocessed.csv")
df_preprocessed.to_csv(preprocessed_csv, index=False, encoding="utf-8")

print(f"✅ Preprocessing complete! Preprocessed audio in: {preprocessed_folder}")
print(f"✅ Updated metadata saved as: {preprocessed_csv}")
