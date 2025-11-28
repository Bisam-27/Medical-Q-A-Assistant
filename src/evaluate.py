import time
import json
from pathlib import Path
from rag_pipeline import MedicalRAG

# Test questions covering different medical areas
TEST_QUESTIONS = [
    "What are the symptoms of diabetes?",
    "How is hypertension treated?",
    "What are the signs of a heart attack?",
    "What causes chest pain?",
    "How is asthma diagnosed?",
    "What is the treatment for pneumonia?",
    "What are the symptoms of stroke?",
    "How is arthritis managed?",
    "What causes kidney stones?",
    "What is a colonoscopy procedure?",
    "How is depression treated?",
    "What are the signs of infection?",
    "What causes headaches?",
    "How is cancer diagnosed?",
    "What are the symptoms of allergies?",
    "What is physical therapy?",
    "How is surgery performed?",
    "What causes back pain?",
    "What are blood tests used for?",
    "How is medication administered?",
    "What are the risks of surgery?",
    "What is preventive care?",
    "How is pain managed?",
    "What causes fatigue?",
    "What are the benefits of exercise?",
    "How is nutrition important?",
    "What causes nausea?",
    "What is rehabilitation?",
    "How are wounds treated?",
    "What causes dizziness?",
    "What is emergency care?",
    "How is recovery monitored?",
    "What are vital signs?"
]

def evaluate_system():
    print("Starting evaluation of Medical RAG system...")
    
    # Initialize RAG system
    rag = MedicalRAG()
    
    results = []
    total_time = 0
    
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"Evaluating question {i}/{len(TEST_QUESTIONS)}: {question}")
        
        start_time = time.time()
        result = rag.answer_question(question)
        response_time = time.time() - start_time
        
        total_time += response_time
        
        # Simple evaluation metrics
        has_answer = len(result['answer']) > 50
        has_sources = len(result['sources']) > 0
        answer_length = len(result['answer'].split())
        
        eval_result = {
            'question': question,
            'answer': result['answer'],
            'response_time': response_time,
            'has_answer': has_answer,
            'has_sources': has_sources,
            'answer_length': answer_length,
            'num_sources': len(result['sources'])
        }
        
        results.append(eval_result)
    
    # Calculate overall metrics
    avg_response_time = total_time / len(TEST_QUESTIONS)
    questions_with_answers = sum(1 for r in results if r['has_answer'])
    questions_with_sources = sum(1 for r in results if r['has_sources'])
    avg_answer_length = sum(r['answer_length'] for r in results) / len(results)
    
    metrics = {
        'total_questions': len(TEST_QUESTIONS),
        'questions_answered': questions_with_answers,
        'success_rate': questions_with_answers / len(TEST_QUESTIONS) * 100,
        'avg_response_time': avg_response_time,
        'avg_answer_length': avg_answer_length,
        'questions_with_sources': questions_with_sources
    }
    
    # Save results
    evaluation_results = {
        'metrics': metrics,
        'detailed_results': results
    }
    
    # Create evaluation directory and save results
    eval_dir = Path(__file__).parent.parent / "evaluation_results"
    eval_dir.mkdir(exist_ok=True)
    
    with open(eval_dir / "evaluation.json", 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Print summary
    print("\\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total Questions: {metrics['total_questions']}")
    print(f"Questions Answered: {metrics['questions_answered']}")
    print(f"Success Rate: {metrics['success_rate']:.1f}%")
    print(f"Average Response Time: {metrics['avg_response_time']:.2f}s")
    print(f"Average Answer Length: {metrics['avg_answer_length']:.0f} words")
    print(f"Questions with Sources: {metrics['questions_with_sources']}")
    print("="*50)
    
    print(f"\\nDetailed results saved to: evaluation_results/evaluation.json")

if __name__ == "__main__":
    evaluate_system()