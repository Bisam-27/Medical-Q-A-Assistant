import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from rag_pipeline import MedicalRAG
import config

st.set_page_config(page_title="Medical Q&A Assistant", page_icon="🏥")

@st.cache_resource
def initialize_system():
    """Initialize the system - create demo data if needed"""
    try:
        # Check if vector store exists
        if not config.VECTOR_STORE_PATH.exists():
            st.info("🔄 Setting up system for first time use...")
            
            # Create demo data for deployment
            demo_chunks = [
                {
                    "text": "Diabetes mellitus symptoms include increased thirst (polydipsia), frequent urination (polyuria), extreme fatigue, blurred vision, slow-healing cuts and wounds, and unexplained weight loss. Type 1 diabetes often develops quickly, while Type 2 diabetes develops gradually over time.",
                    "specialty": "Endocrinology", 
                    "description": "Diabetes mellitus symptoms and clinical presentation",
                    "source_id": 1
                },
                {
                    "text": "Hypertension treatment involves lifestyle modifications including reduced sodium intake, regular exercise, weight management, and stress reduction. Medications include ACE inhibitors, diuretics, beta-blockers, and calcium channel blockers. Blood pressure should be monitored regularly.",
                    "specialty": "Cardiology",
                    "description": "Hypertension management and antihypertensive medications", 
                    "source_id": 2
                },
                {
                    "text": "Myocardial infarction (heart attack) symptoms include severe chest pain or pressure, shortness of breath, nausea, sweating, and pain radiating to the left arm, neck, or jaw. Call emergency services immediately. Treatment includes medications to restore blood flow.",
                    "specialty": "Cardiology",
                    "description": "Acute myocardial infarction signs, symptoms and emergency treatment",
                    "source_id": 3
                },
                {
                    "text": "Asthma diagnosis involves medical history, physical examination, and pulmonary function tests including spirometry. Peak flow monitoring helps assess lung function. Chest X-rays may be performed to rule out other conditions.",
                    "specialty": "Pulmonology",
                    "description": "Asthma diagnostic procedures and pulmonary function testing",
                    "source_id": 4
                },
                {
                    "text": "Pneumonia treatment depends on the causative organism. Bacterial pneumonia is treated with antibiotics, while viral pneumonia is managed with supportive care. Severe cases may require hospitalization and oxygen therapy.",
                    "specialty": "Internal Medicine",
                    "description": "Pneumonia treatment protocols and antimicrobial therapy",
                    "source_id": 5
                }
            ]
            
            # Create directories
            config.DATA_DIR.mkdir(parents=True, exist_ok=True)
            (config.DATA_DIR / "processed").mkdir(exist_ok=True)
            
            # Save demo chunks
            import json
            with open(config.PROCESSED_DATA_PATH, 'w', encoding='utf-8') as f:
                json.dump(demo_chunks, f, indent=2)
            
            # Create vector store
            from build_vector_store import VectorStore
            store = VectorStore()
            store.build_from_chunks(demo_chunks)
            store.save(config.VECTOR_STORE_PATH)
            
            st.success("✅ Demo system initialized!")
        
        # Load the RAG system
        return MedicalRAG()
        
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        st.error("Please check your API key configuration.")
        return None

def main():
    st.title("🏥 Medical Q&A Assistant")
    st.write("Ask medical questions and get evidence-based answers")
    
    # Initialize system
    rag = initialize_system()
    
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
