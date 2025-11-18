# 🧠 StudyBuddy – Smart AI Learning Assistant

> **StudyBuddy** is an AI-powered web application that helps students learn smarter and faster.  
> Upload PDFs, get instant **summaries**, generate **quizzes**, and even receive **AI-based answer evaluations** — all in one elegant interface.

![Flask](https://img.shields.io/badge/Backend-Flask-blue?style=flat-square)
![LangChain](https://img.shields.io/badge/AI-LangChain-green?style=flat-square)
![HuggingFace](https://img.shields.io/badge/Model-HuggingFace-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 🚀 Features

✅ **PDF Summarization**  
Upload lengthy study materials or notes and receive clear, concise summaries focused on key learning points.

✅ **Quiz Generation**  
Convert any text or passage into interactive, multiple-choice quiz questions that test comprehension and recall.

✅ **Answer Evaluation**  
AI evaluates student responses with a score (1–10) and personalized feedback (Coming soon to UI).

✅ **Dark Mode UI**  
Beautiful, responsive, and professional dark-themed interface built with glassmorphism and live shimmer effects.

✅ **Chunked AI Processing**  
Supports large PDFs by splitting text into smaller chunks, ensuring accurate summarization without truncation.

---

## 🧩 System Architecture

```
Frontend (HTML, CSS, JS)
        ↓
Flask Backend (Python)
        ↓
LangChain Framework
        ↓
HuggingFace / OpenRouter (Mistral-7B)
        ↓
AI Output → Summaries, Quizzes, Evaluations
```

- **Frontend:** Sends user inputs (text or files) via Fetch API  
- **Flask Backend:** Handles requests and runs AI pipelines  
- **LangChain:** Manages prompts and model interactions  
- **OpenRouter (Mistral-7B):** Generates intelligent text responses  
- **PyPDFLoader:** Extracts text from uploaded PDFs  

---

## 🧠 Tech Stack

| Layer | Technology | Purpose |
|-------|-------------|----------|
| **Frontend** | HTML5, CSS3, Vanilla JS | Interactive dark theme UI |
| **Backend** | Flask (Python), Flask-CORS | API routing and data flow |
| **AI Framework** | LangChain + HuggingFacePipeline | Prompt chaining and model execution |
| **LLM Model** | Mistral-7B via OpenRouter | High-quality text generation |
| **Utilities** | PyPDFLoader | Extracts text from PDFs |

---

## ⚙️ Installation & Setup

Follow these steps to run StudyBuddy locally:

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/StudyBuddy.git
cd StudyBuddy
```

### 2️⃣ Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # (Mac/Linux)
venv\Scripts\activate      # (Windows)
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Flask server
```bash
python app.py
```

### 5️⃣ Open the app
Visit 👉 **http://127.0.0.1:5000** in your browser.

---

## 🧬 API Endpoints

| Endpoint | Method | Description |
|-----------|--------|-------------|
| `/` | GET | Serves the StudyBuddy frontend |
| `/summarize` | POST | Upload a PDF to generate a summary |
| `/generate-quiz` | POST | Send text and get quiz questions |
| `/evaluate-answer` | POST | Evaluate a student’s written answer |

---

## 📄 Example API Usage

**Generate Quiz Example (cURL):**
```bash
curl -X POST http://127.0.0.1:5000/generate-quiz \
  -H "Content-Type: application/json" \
  -d '{"context": "Photosynthesis is the process by which plants convert sunlight into energy.", "num_questions": 3}'
```

**Response:**
```json
{
  "quiz": "Q: What is the main purpose of photosynthesis?\nA. ...\nB. ...\nC. ...\nD. ...\nAnswer: B"
}
```

---

## 🌍 Environment Variables

To use **OpenRouter** (for higher quality summarization & quiz generation):

| Variable | Description |
|-----------|--------------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key (from [openrouter.ai](https://openrouter.ai)) |

If not set, the app defaults to a built-in placeholder key.

---



## 🧩 Future Enhancements

- 🗣️ Voice-based summarization  
- 📊 Student analytics dashboard  
- ☁️ Cloud deployment (Render / AWS / Vercel)  
- 🧾 Flashcard export & note storage  
- 💬 Chat-style interaction with AI  

---



## 📜 License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.

---

### 💡 *“Empowering students with AI for smarter, faster learning.”*
