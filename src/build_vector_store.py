import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import config

class VectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.vectors = None
        self.documents = []
    
    def build_from_chunks(self, chunks):
        """Build vector store from text chunks"""
        print("Building vector store...")
        
        texts = [chunk['text'] for chunk in chunks]
        self.documents = chunks
        
        # Create TF-IDF vectors
        self.vectors = self.vectorizer.fit_transform(texts)
        
        print(f"Built vector store with {len(chunks)} documents")
    
    def search(self, query, k=5):
        """Search for similar documents"""
        if self.vectors is None:
            return []
        
        # Transform query to vector
        query_vec = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vec, self.vectors).flatten()
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Only return relevant results
                results.append({
                    'document': self.documents[idx],
                    'score': similarities[idx]
                })
        
        return results
    
    def save(self, path):
        """Save vector store to disk"""
        path.mkdir(parents=True, exist_ok=True)
        
        with open(path / 'vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(path / 'vectors.pkl', 'wb') as f:
            pickle.dump(self.vectors, f)
        
        with open(path / 'documents.json', 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, indent=2)
    
    @classmethod
    def load(cls, path):
        """Load vector store from disk"""
        store = cls()
        
        with open(path / 'vectorizer.pkl', 'rb') as f:
            store.vectorizer = pickle.load(f)
        
        with open(path / 'vectors.pkl', 'rb') as f:
            store.vectors = pickle.load(f)
        
        with open(path / 'documents.json', 'r', encoding='utf-8') as f:
            store.documents = json.load(f)
        
        return store

def main():
    # Load processed chunks
    with open(config.PROCESSED_DATA_PATH, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Build and save vector store
    store = VectorStore()
    store.build_from_chunks(chunks)
    store.save(config.VECTOR_STORE_PATH)
    
    # Test search
    results = store.search("diabetes symptoms", k=3)
    print(f"\nTest search returned {len(results)} results")

if __name__ == "__main__":
    main()
