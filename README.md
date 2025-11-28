# Medical RAG Q&A System

A simple medical question-answering system using Retrieval-Augmented Generation (RAG) with Google's Gemini API.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download dataset:**
   - Download the medical transcriptions dataset from [Kaggle](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)
   - Place `mtsamples.csv` in `data/raw/` folder

3. **Configure API key:**
   - Get a free Gemini API key from [Google AI Studio](https://ai.google.dev/)
   - Update `GEMINI_API_KEY` in `src/config.py`

## Usage

1. **Preprocess data:**
   ```bash
   python src/preprocess.py
   ```

2. **Build vector store:**
   ```bash
   python src/build_vector_store.py
   ```

3. **Run the web app:**
   ```bash
   streamlit run src/app.py
   ```

4. **Evaluate system:**
   ```bash
   python src/evaluate.py
   ```

## Features

- Medical question answering with source citations
- TF-IDF based document retrieval
- Simple Streamlit web interface
- Evaluation on 33 medical queries

## Disclaimer

This system is for educational purposes only. Always consult healthcare professionals for medical advice.
