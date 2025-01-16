from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging
from datetime import datetime
import threading
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# Constants
MODEL_NAME = "google/flan-t5-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 256  # Shorter for more focused responses

# Global variables
model_lock = threading.Lock()
model: Optional[AutoModelForSeq2SeqLM] = None
tokenizer: Optional[AutoTokenizer] = None

# Simplified subject definitions with explicit prompts
SUBJECTS = {
    'math': {
        'keywords': ['solve', 'calculate', '+', '-', '*', '/', '=', 'equation'],
        'prompt_template': """
Solve this math problem step by step:
Question: {question}
Rules:
1. Show each calculation step
2. Use numbers and mathematical symbols
3. Give the final answer clearly
Solve it now:"""
    },
    'science': {
        'keywords': ['explain', 'what is', 'how does', 'why', 'science', 'biology', 'chemistry', 'physics'],
        'prompt_template': """
Explain this science concept:
Question: {question}
Rules:
1. Start with a simple definition
2. Explain the main process
3. Give a real-world example
Explain it now:"""
    },
    'general': {
        'keywords': ['what', 'where', 'when', 'who', 'capital', 'describe'],
        'prompt_template': """
Answer this question directly and accurately:
Question: {question}
Rules:
1. Give the main answer first
2. Add 1-2 relevant facts
3. Keep it concise
Answer it now:"""
    }
}

def initialize_model() -> bool:
    """Initialize model with minimal settings"""
    global model, tokenizer
    try:
        with model_lock:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
            model.eval()
        return True
    except Exception as e:
        logging.error(f"Model initialization error: {e}")
        return False

def detect_subject(question: str) -> Tuple[str, dict]:
    """Simple and efficient subject detection"""
    question_lower = question.lower()
    
    # Check math first (highest priority)
    if any(op in question for op in ['+', '-', '*', '/', '=']) or \
       any(term in question_lower for term in ['solve', 'calculate', 'equation']):
        return 'math', SUBJECTS['math']
    
    # Check science
    if any(term in question_lower for term in ['explain', 'how', 'why', 'what is']) and \
       any(term in question_lower for term in ['science', 'biology', 'chemistry', 'physics']):
        return 'science', SUBJECTS['science']
    
    # Default to general
    return 'general', SUBJECTS['general']

def generate_response(question: str, subject_info: dict) -> Optional[str]:
    """Generate response with focused prompt"""
    try:
        with model_lock:
            # Create focused prompt
            prompt = subject_info['prompt_template'].format(question=question)
            
            # Generate response
            inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_LENGTH, truncation=True).to(DEVICE)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=MAX_LENGTH,
                    num_return_sequences=1,
                    temperature=0.3,  # Low temperature for focused responses
                    do_sample=False,  # Deterministic output
                    length_penalty=1.0
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean response
            response = response.replace(prompt, '').strip()
            response = response.replace('Answer:', '').replace('Solution:', '').strip()
            
            # Format response based on subject
            if 'math' in subject_info['prompt_template']:
                lines = [line.strip() for line in response.split('\n') if line.strip()]
                response = '\n'.join(f"Step {i+1}: {line}" if not line.startswith('Step') else line 
                                   for i, line in enumerate(lines))
            
            return response

    except Exception as e:
        logging.error(f"Generation error: {e}")
        return None

@app.route('/chat', methods=['POST'])
def chat():
    """Simplified chat endpoint"""
    try:
        question = request.json.get('message', '').strip()
        if not question:
            return jsonify({"error": "Empty question"}), 400

        subject, subject_info = detect_subject(question)
        response = generate_response(question, subject_info)
        
        if not response:
            return jsonify({"error": "Failed to generate response"}), 500

        return jsonify({
            "response": response,
            "subject": subject,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logging.error(f"API error: {e}")
        return jsonify({"error": "Server error"}), 500

if __name__ == '__main__':
    if initialize_model():
        app.run(host='0.0.0.0', port=5000)
    else:
        logging.error("Model initialization failed")