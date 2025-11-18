import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader # <-- UPDATED IMPORT
from langchain_core.prompts import PromptTemplate # <-- UPDATED IMPORT
from langchain_classic.chains.llm import LLMChain  # DEPRECATED, but matches usage
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import logging
from math import ceil
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. Initialize Flask App and CORS ---
# CORS is required to allow your index.html (on a 'file://' domain)
# to talk to your server (on 'http://127.0.0.1:5000')
app = Flask(__name__)
CORS(app)

# --- 2. Initialize the AI Model (LLM) ---
# We use a single, smaller model (flan-t5-small) to handle all tasks.
# This saves memory. It will download the model the first time you run this.
logger.info("Initializing Hugging Face Pipeline...")
model_id = "google/flan-t5-base"  # UPGRADED FROM flan-t5-small
hf_pipeline = pipeline(
    "text2text-generation",
    model=model_id,
    max_new_tokens=512,
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)
logger.info("Hugging Face Pipeline initialized successfully.")

# --- 3. Create LangChain Chains for each task ---
# a) Summarization Chain using a prompt
summarize_prompt = PromptTemplate(
    input_variables=["context"],
    template="""
    Please provide a detailed and comprehensive summary of the following text. Focus on the main ideas, important facts, and key points, while remaining concise and clear for a student to study from.
    
    TEXT:
    {context}
    
    SUMMARY:
    """
)
summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)

# b) Quiz Generation Chain
quiz_prompt = PromptTemplate(
    input_variables=["context"],
    template="""
    You are a helpful study assistant. Based on the following text, generate 3 unique, high-quality multiple-choice questions that test comprehension and recall. Each question should have 4 distinct options (A, B, C, D) and indicate the correct answer letter only. Questions should not be trivial or repetitive.

    Format for each question:
      Q: ...
      A. ...
      B. ...
      C. ...
      D. ...
      Answer: [A/B/C/D]

    TEXT:
    {context}
    
    QUIZ:
    """
)
quiz_chain = LLMChain(llm=llm, prompt=quiz_prompt)

# c) Answer Evaluation Chain
eval_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="""
    Evaluate the student's answer to the following question.
    Provide a "Score" from 1 to 10 and "Feedback" explaining your reasoning.
    Format the output exactly like this:

    Score: [Your Score]/10
    Feedback: [Your Feedback]

    QUESTION:
    {question}

    STUDENT ANSWER:
    {answer}

    EVALUATION:
    """
)
eval_chain = LLMChain(llm=llm, prompt=eval_prompt)

# --- 4. Define API Endpoints ---

@app.route("/")
def home():
    # Serve the frontend index.html
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "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")
OPENROUTER_COMPLETION_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "mistralai/mistral-7b-instruct"

def call_openrouter(messages, model=DEFAULT_MODEL, max_tokens=512):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    response = requests.post(OPENROUTER_COMPLETION_URL, headers=headers, json=body)
    if response.status_code == 200:
        data = response.json()
        # Assumes OpenAI-compatible with 'choices'
        return data['choices'][0]['message']['content']
    else:
        raise Exception(f"OpenRouter API error: {response.status_code} {response.text}")

def chunk_text(text, chunk_size=1500, overlap=200):
    words = text.split()
    n = len(words)
    if n <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start = end - overlap  # overlap for context
        if start < 0: start = 0
    return chunks

@app.route("/summarize", methods=["POST"])
def summarize_pdf():
    """
    Handles PDF file upload, extracts text, and returns a summary.
    """
    logger.info("Received request for /summarize")
    if 'file' not in request.files:
        logger.warning("No file part in request")
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '' or not file.filename.endswith('.pdf'):
        logger.warning("Invalid file (empty or not PDF)")
        return jsonify({"error": "Invalid file. Must be a .pdf"}), 400

    try:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_pdf_path = os.path.join(temp_dir, file.filename)
        file.save(temp_pdf_path)
        logger.info(f"File saved to {temp_pdf_path}")

        # Load text from PDF
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()
        
        if not docs:
            logger.error("Could not extract text from PDF")
            return jsonify({"error": "Could not extract text from PDF"}), 500
        
        logger.info("Running summarization chain with chunking...")
        # NEW CHUNKING LOGIC
        all_text = "\n".join([doc.page_content for doc in docs])
        chunks = chunk_text(all_text)
        summaries = []
        for chunk in chunks:
            summary_prompt = [{
                "role": "system",
                "content": "You are an expert educational assistant. Summarize the following content for study revision, keeping key points and useful details concisely clear for a student."
            }, {
                "role": "user",
                "content": chunk
            }]
            resp = call_openrouter(summary_prompt)
            summaries.append(resp.strip())
        final = summaries[0] if len(summaries) == 1 else call_openrouter([
          {"role": "system", "content": "You are an expert educational summarizer. Concisely merge and summarize the following points for maximum clarity."},
          {"role": "user", "content": "\n\n".join(summaries)}
        ])

        os.remove(temp_pdf_path)
        
        logger.info("Summary generated successfully")
        return jsonify({"summary": final})

    except Exception as e:
        logger.error(f"Error in /summarize: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/generate-quiz", methods=["POST"])
def generate_quiz():
    """
    Generates a quiz from a given block of text.
    """
    logger.info("Received request for /generate-quiz")
    data = request.json
    if not data or "context" not in data or not data["context"].strip():
        logger.warning("No context provided for quiz")
        return jsonify({"error": "No context provided"}), 400
    num_questions = 3
    if "num_questions" in data:
        try:
            n = int(data["num_questions"])
            if 1 <= n <= 10:
                num_questions = n
        except Exception:
            pass
    try:
        n = max(min(int(data.get("num_questions", 3)), 10), 1)
        chunks = chunk_text(data["context"])
        all_questions = []
        for chunk in chunks:
            quiz_prompt = [{
                "role": "system",
                "content": f"You are a world-class quiz generator for students. Based ONLY on the following, write {n} unique, medium-difficulty, multiple-choice questions with 4 options each (A, B, C, D), marking the correct option like 'Answer: C'."
            }, {
                "role": "user",
                "content": chunk
            }]
            resp = call_openrouter(quiz_prompt, max_tokens=400+70*n)
            all_questions.append(resp.strip())
        return jsonify({"quiz": '\n\n'.join(all_questions)})
    except Exception as e:
        logger.error(f"Error in /generate-quiz: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/evaluate-answer", methods=["POST"])
def evaluate_answer():
    """
    Evaluates a student's answer to a question.
    """
    logger.info("Received request for /evaluate-answer")
    data = request.json
    if not data or "question" not in data or "answer" not in data:
        logger.warning("Missing question or answer")
        return jsonify({"error": "Missing 'question' or 'answer'"}), 400
        
    try:
        question = data["question"]
        answer = data["answer"]
        
        logger.info("Running evaluation chain...")
        # Run the evaluation chain
        eval_result_text = eval_chain.run(question=question, answer=answer)
        
        logger.info("Evaluation generated successfully")
        # We return the raw text. The frontend will parse it.
        return jsonify({"evaluation_text": eval_result_text})

    except Exception as e:
        logger.error(f"Error in /evaluate-answer: {e}")
        return jsonify({"error": str(e)}), 500

# --- 5. Run the App ---
if __name__ == "__main__":
    logger.info("Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)



