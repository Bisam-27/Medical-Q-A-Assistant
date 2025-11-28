import os
from pathlib import Path

# API Key for Gemini - try Streamlit secrets first, then fallback
try:
    import streamlit as st
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    GEMINI_API_KEY = "Your_API_KEY"

# Model configuration
MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.3
MAX_TOKENS = 1024

# Data paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "mtsamples.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "chunks.json"
VECTOR_STORE_PATH = DATA_DIR / "vector_store"

# Processing settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 5

def get_api_key():
    return GEMINI_API_KEY
