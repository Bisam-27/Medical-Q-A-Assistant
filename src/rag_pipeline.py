import google.generativeai as genai
from build_vector_store import VectorStore
import config

class MedicalRAG:
    def __init__(self):
        # Configure Gemini
        genai.configure(api_key=config.get_api_key())
        self.model = genai.GenerativeModel(config.MODEL_NAME)
        
        # Load vector store
        self.vector_store = VectorStore.load(config.VECTOR_STORE_PATH)
        
        print("Medical RAG system initialized")
    
    def answer_question(self, question):
        """Answer a medical question using RAG"""
        
        # Retrieve relevant documents
        results = self.vector_store.search(question, k=config.TOP_K_RESULTS)
        
        if not results:
            return {
                'answer': "I couldn't find relevant medical information for your question.",
                'sources': []
            }
        
        # Prepare context from retrieved documents
        context = "\\n\\n".join([
            f"Medical Specialty: {doc['document']['specialty']}\\n{doc['document']['text']}"
            for doc in results[:3]  # Use top 3 results
        ])
        
        # Create prompt
        prompt = f"""You are a medical information assistant. Based on the following medical context, provide a helpful answer.

Context:
{context}

Question: {question}

Please provide an accurate answer based only on the context provided. If the context doesn't contain enough information, say so clearly."""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=config.TEMPERATURE,
                    max_output_tokens=config.MAX_TOKENS,
                )
            )
            
            answer = response.text
            sources = [r['document'] for r in results[:3]]
            
            return {
                'answer': answer,
                'sources': sources
            }
            
        except Exception as e:
            return {
                'answer': f"Error generating response: {str(e)}",
                'sources': []
            }

def main():
    # Test the RAG system
    rag = MedicalRAG()
    
    test_questions = [
        "What are the symptoms of diabetes?",
        "How is hypertension treated?",
    ]
    
    for question in test_questions:
        print(f"\\nQuestion: {question}")
        result = rag.answer_question(question)
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Sources: {len(result['sources'])} medical documents")

if __name__ == "__main__":
    main()