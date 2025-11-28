import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from rag_pipeline import MedicalRAG
import config

st.set_page_config(page_title="Medical Q&A Assistant", page_icon="🏥")

@st.cache_resource
def load_rag_system():
    # Check if data exists, if not show setup instructions
    try:
        if not config.VECTOR_STORE_PATH.exists():
            st.error("⚠️ Vector store not found!")
            st.info("Please run the following commands first:")
            st.code("""
# In the project directory, run:
python src/preprocess.py
python src/build_vector_store.py
            """)
            return None
        
        # Try to load the RAG system
        return MedicalRAG()
        
    except Exception as e:
        st.error(f"Error setting up system: {str(e)}")
        return None

def main():
    st.title("🏥 Medical Q&A Assistant")
    st.write("Ask medical questions and get evidence-based answers")
    
    # Load RAG system
    rag = load_rag_system()
    
    if rag is None:
        st.stop()  # Stop execution if system couldn't load
    
    # Question input
    question = st.text_area(
        "Enter your medical question:",
        placeholder="e.g., What are the symptoms of diabetes?",
        height=100
    )
    
    if st.button("Get Answer", type="primary"):
        if question.strip():
            with st.spinner("Searching medical literature..."):
                result = rag.answer_question(question)
            
            # Display answer
            st.subheader("Answer")
            st.write(result['answer'])
            
            # Display sources
            if result['sources']:
                st.subheader("Sources")
                for i, source in enumerate(result['sources'], 1):
                    with st.expander(f"Source {i}: {source['specialty']}"):
                        st.write(f"**Description:** {source['description']}")
                        st.write(f"**Content:** {source['text'][:300]}...")
        else:
            st.warning("Please enter a question.")
    
    # Disclaimer
    st.sidebar.markdown("""
    ### ⚠️ Medical Disclaimer
    
    This tool is for informational purposes only. 
    Always consult with healthcare professionals 
    for medical advice.
    """)
    
    # Example questions
    st.sidebar.markdown("""
    ### 💡 Example Questions
    
    - What are the symptoms of diabetes?
    - How is hypertension treated?
    - What causes chest pain?
    - What is a colonoscopy procedure?
    """)

if __name__ == "__main__":
    main()
