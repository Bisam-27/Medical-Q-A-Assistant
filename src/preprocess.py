import pandas as pd
import json
import re
from pathlib import Path
import config

def clean_text(text):
    """Basic text cleaning"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

def create_chunks(text, metadata, chunk_size=500, overlap=50):
    """Split text into chunks"""
    chunks = []
    words = text.split()
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        
        chunk = {
            'text': chunk_text,
            'specialty': metadata.get('medical_specialty', ''),
            'description': metadata.get('description', ''),
            'source_id': metadata.get('source_id', 0)
        }
        chunks.append(chunk)
    
    return chunks

def main():
    print("Loading medical data...")
    df = pd.read_csv(config.RAW_DATA_PATH)
    
    all_chunks = []
    for idx, row in df.iterrows():
        text = clean_text(row.get('transcription', ''))
        if len(text) < 50:  # Skip very short texts
            continue
            
        metadata = {
            'medical_specialty': clean_text(row.get('medical_specialty', '')),
            'description': clean_text(row.get('description', '')),
            'source_id': idx
        }
        
        chunks = create_chunks(text, metadata)
        all_chunks.extend(chunks)
    
    # Save chunks
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    (config.DATA_DIR / "processed").mkdir(exist_ok=True)
    
    with open(config.PROCESSED_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)
    
    print(f"Created {len(all_chunks)} chunks from {len(df)} documents")

if __name__ == "__main__":
    main()