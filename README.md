# GenAI Offer Generator

**American Express Hackathon 2025**

AI-powered system for personalized credit card offers using behavioral analysis, RAG, and LLM.

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python api.py
python api_example.py
```

API Docs: http://localhost:8000/docs

---

## 📡 API

**Input** (ML Model Output):
```json
{
  "ml_output": {
    "user_id": "USR001",
    "offer_flag": true,
    "domain": "Travel",
    "confidence_score": 0.85
  }
}
```

**Output**: Personalized offer with reasoning, behavioral analysis, and email HTML.

---

## 🏗️ Architecture

```
ML Output → User Retrieval → Behavioral Analysis → Policy RAG → LLM → Offer
```

---

## ⚙️ Config

Edit `config.py`:
```python
OPENROUTER_API_KEY = "your-key"
MIN_CONFIDENCE_SCORE = 0.7
```

---

## 🔧 Stack

FastAPI • OpenRouter • ChromaDB • Sentence Transformers

---

## 📊 Endpoints

- `POST /generate-offer` - Single user
- `POST /batch-generate-offers` - Batch
- `GET /health` - Status

---

## 📁 Key Scripts

**Core Components**:
- `api.py` - FastAPI server with REST endpoints
- `genai_reasoner.py` - Main orchestrator coordinating all components
- `root_cause_analyzer.py` - Analyzes user transaction history (hybrid: rule-based + LLM reasoning)
- `offer_generator.py` - Generates personalized offers based on user insights, ML output, and policies
- `rag_system.py` - Policy retrieval using vector search (ChromaDB)
- `user_data_store.py` - User data management and retrieval

**Utilities**:
- `config.py` - Configuration settings
- `sample_data.py` - Demo user data generator
- `api_example.py` - API client examples
- `demo.py` - Interactive demonstration
- `simple_example.py` - Minimal usage example
