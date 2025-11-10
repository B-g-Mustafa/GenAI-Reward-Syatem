# GenAI Offer Generator

**American Express Hackathon 2025**

AI-powered system for personalized credit card offers using behavioral analysis, RAG, and LLM.

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Pull and run from Docker Hub (with default key)
docker pull yourusername/amex-offer-generator:latest
docker run -d -p 9000:9000 --name amex-app yourusername/amex-offer-generator:latest

# Or with custom OpenRouter API key
docker run -d -p 9000:9000 -e OPENROUTER_API_KEY="your-api-key-here" --name amex-app yourusername/amex-offer-generator:latest
```

### Option 2: Local Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
python api.py
```

**API Docs**: http://localhost:9000/docs

---

## 📡 API

### POST `/generate-offer`

**New Batch Input Format** (Supports multiple users):
```json
{
  "ml_output": {
    "total_users": 2,
    "predictions": [
      {
        "user_id": "USER_000000",
        "offer_flag": 1,
        "domain": "Electronics",
        "confidence_score": 0.9588919271748322,
        "pred_version": "v1",
        "prediction_timestamp": "2025-10-28T10:28:16.506594"
      },
      {
        "user_id": "USER_000001",
        "offer_flag": 1,
        "domain": "Travel",
        "confidence_score": 0.8234567890123456,
        "pred_version": "v1",
        "prediction_timestamp": "2025-10-28T10:28:16.506594"
      }
    ],
    "processing_time_seconds": 0.037884
  },
  "use_llm_insights": true
}
```

**Field Descriptions**:
- `predictions` - Array of ML predictions (can be single or multiple users)
- `user_id` - Card member identifier (must exist in PostgreSQL database)
- `offer_flag` - 1 = generate offer, 0 = skip user
- `domain` - Target category: Travel, Dining, Electronics, Business Services, Retail, Entertainment
- `confidence_score` - ML model confidence (0-1)
- `use_llm_insights` - Enable/disable LLM-enhanced behavioral analysis (default: true)

**Output**: Array of offer responses with reasoning, behavioral analysis, and email HTML.

**Example Output**:
```json
[
  {
    "success": true,
    "user_id": "USER_000000",
    "timestamp": "2025-10-30T12:30:45.123456",
    "recommendation": "offer_generated",
    "ml_recommendation": { ... },
    "behavioral_analysis": {
      "insights": ["User's Electronics spending has increased...", ...],
      "spending_trends": { ... },
      "engagement_metrics": { ... }
    },
    "applied_policies": [ ... ],
    "offer": {
      "success": true,
      "offer_title": "3X Points on Electronics",
      "offer_value": "3X Points",
      "offer_description": "...",
      "terms_and_conditions": "...",
      "call_to_action": "...",
      "offer_type": "points_multiplier",
      "expiration_days": 45,
      "minimum_spend": 500,
      "featured_merchants": [
        {
          "merchant_name": "Best Buy",
          "merchant_category": "Electronics",
          "merchant_id": "MERCH_BESTBUY_901",
          "merchant_type": "both"
        },
        {
          "merchant_name": "Apple Store",
          "merchant_category": "Electronics",
          "merchant_id": "MERCH_APPLE_1001",
          "merchant_type": "both"
        }
      ]
    },
    "email_html": "<html>...</html>"
  }
]
```

---

## 🏗️ Architecture

```
ML Predictions → PostgreSQL User Retrieval → Behavioral Analysis (Rule-Based + LLM) 
                                                      ↓
                                          Policy RAG (ChromaDB)
                                                      ↓
                                          LLM Offer Generation
                                                      ↓
                                     Personalized Offer + Email HTML
```

**Pipeline Steps**:
1. **User Data Retrieval** - Fetch user profile, spending, and engagement data from PostgreSQL
2. **Behavioral Analysis** - Hybrid approach combining:
   - Rule-based insights (spending trends, engagement patterns)
   - LLM-enhanced insights (psychological patterns, root cause reasoning)
3. **Policy Retrieval** - RAG system fetches relevant policies from ChromaDB
4. **Offer Generation** - LLM creates personalized, policy-compliant offers
5. **Email Formatting** - Generate ready-to-send HTML email

---

## 💾 Database Schema

### User Data (PostgreSQL - Flat Structure)

User data is stored with **flat columns** (no nested JSON):

**Profile Fields**: `user_id`, `name`, `age`, `gender`, `location`, `segment`, `tenure_years`, `card_type`, `annual_fee`, `credit_limit`, `persona`, `customer_lifecycle_stage`, `churn_risk_score`

**Spending Metrics**: `total_transactions_12m`, `total_spend_12m`, `avg_transaction_amount`, `recency_days`, `frequency_30d`, `monetary_30d`, `spend_change_pct_30d`, `spend_change_pct_90d`

**Engagement Metrics**: `email_open_rate`, `total_app_opens_90d`, `offer_views_90d`, `offer_clicks_90d`, `avg_session_duration_sec`, `engagement_change_pct_30d`

**Offer History**: `offers_shown_6m`, `offers_accepted_6m`, `historical_acceptance_rate`, `days_since_last_offer`, `last_offer_domain`, `last_offer_date`

**Preferences**: `preferred_domain_1`, `preferred_domain_2`

> **Note**: The system automatically retrieves user data from PostgreSQL using `user_id` - you don't need to provide full user profiles in the API request.

---

## ⚙️ Configuration

Edit `config.py`:
```python
# OpenRouter LLM Configuration
OPENROUTER_API_KEY = "your-key-here"
OPENROUTER_MODEL = "meta-llama/llama-3.1-70b-instruct"

# Business Rules
MIN_CONFIDENCE_SCORE = 0.7
MAX_OFFERS_PER_MONTH = 4
DOMAINS = ["Travel", "Dining", "Electronics", "Business Services", "Retail", "Entertainment"]

# PostgreSQL Connection (configured in user_data_store.py)
DB_CONFIG = {
    'host': 'your-host.neon.tech',
    'database': 'neondb',
    'user': 'neondb_owner',
    'password': 'your-password',
    'sslmode': 'require'
}
```

---

## 🔧 Tech Stack

- **API**: FastAPI with async support
- **LLM**: OpenRouter (Llama 3.1 70B)
- **Vector DB**: ChromaDB for policy retrieval
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Database**: PostgreSQL (Neon)
- **Python**: 3.8+

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check and config status |
| `/generate-offer` | POST | Generate offers for one or more users |
| `/policies` | GET | List all loaded policies |
| `/add-policy` | POST | Add custom policy to system |
| `/docs` | GET | Interactive API documentation (Swagger) |
| `/redoc` | GET | API documentation (ReDoc) |

---

## 📁 Project Structure

```
amex-hackathon/
├── api.py                      # FastAPI server with REST endpoints
├── genai_reasoner.py          # Main orchestrator coordinating all components
├── root_cause_analyzer.py     # Behavioral analysis (rule-based + LLM)
├── offer_generator.py         # LLM-powered offer generation
├── rag_system.py              # Policy retrieval using ChromaDB
├── user_data_store.py         # PostgreSQL user data management
├── config.py                  # Configuration settings
├── sample_data.py             # Sample data for testing
├── demo.py                    # Interactive demonstration
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── COLUMN_FORMAT_UPDATE.md    # Database schema documentation
├── chroma_db/                 # ChromaDB vector store
└── output/                    # Generated offers (JSON + HTML)
```

### Core Components

**`api.py`** - FastAPI REST API Server
- Accepts ML predictions in batch format
- Retrieves user data from PostgreSQL
- Processes offers asynchronously
- Exports offers to JSON and HTML

**`genai_reasoner.py`** - System Orchestrator
- Coordinates all components
- Manages pipeline flow
- Combines behavioral analysis with policy retrieval
- Configurable LLM insights mode

**`root_cause_analyzer.py`** - Behavioral Analysis Engine
- **Hybrid Approach**: Rule-based + LLM-enhanced insights
- Analyzes spending patterns and trends
- Evaluates engagement metrics
- Reviews offer history and preferences
- Generates contextual insights about user behavior

**`offer_generator.py`** - Personalized Offer Creator
- Uses LLM to generate tailored offers
- Ensures policy compliance
- Creates compelling copy and CTAs
- Formats offers for email delivery

**`rag_system.py`** - Policy Retrieval System
- Vector-based policy search using ChromaDB
- Domain-specific policy filtering
- Compliance and eligibility rules
- Merchant requirements and restrictions

**`user_data_store.py`** - Database Interface
- PostgreSQL connection management
- User data retrieval with flat column structure
- Backward compatibility with nested formats
- Connection pooling and error handling

---

## 🎯 Key Features

### 1. **Hybrid Behavioral Analysis**
Combines fast rule-based insights with deep LLM reasoning:
- Rule-based: Quantitative metrics (spending trends, engagement levels)
- LLM-enhanced: Qualitative insights (psychological patterns, motivations)
- Configurable per request with `use_llm_insights` flag

### 2. **Policy-Aware Offer Generation**
- RAG system retrieves relevant policies from vector store
- LLM generates offers that comply with all policies
- Automatic validation of terms and conditions

### 3. **Batch Processing**
- Process multiple users in a single API call
- Asynchronous offer generation
- Background task for file exports

### 4. **Rich User Profiles**
- 40+ user attributes from PostgreSQL
- Comprehensive spending and engagement metrics
- Historical offer performance data
- Persona and lifecycle stage information

### 5. **Merchant-Level Personalization**
- Partner merchant data integrated into RAG system
- 50+ merchants across 6 domains
- Offers include specific merchant names and IDs
- Dynamic merchant selection based on user behavior

### 6. **Email-Ready Output**
- HTML email templates included
- Personalized content and styling
- Ready for delivery via email service

---

## 🚦 Usage Examples

### Example 1: Generate Single Offer

```bash
curl -X POST "http://localhost:9000/generate-offer" \
  -H "Content-Type: application/json" \
  -d '{
    "ml_output": {
      "predictions": [{
        "user_id": "USER_000000",
        "offer_flag": 1,
        "domain": "Travel",
        "confidence_score": 0.85
      }]
    },
    "use_llm_insights": true
  }'
```

### Example 2: Batch Processing

```bash
curl -X POST "http://localhost:9000/generate-offer" \
  -H "Content-Type: application/json" \
  -d '{
    "ml_output": {
      "total_users": 3,
      "predictions": [
        {"user_id": "USER_001", "offer_flag": 1, "domain": "Travel", "confidence_score": 0.85},
        {"user_id": "USER_002", "offer_flag": 1, "domain": "Dining", "confidence_score": 0.92},
        {"user_id": "USER_003", "offer_flag": 1, "domain": "Electronics", "confidence_score": 0.78}
      ]
    },
    "use_llm_insights": true
  }'
```

### Example 3: Rule-Based Only (Faster)

```bash
curl -X POST "http://localhost:9000/generate-offer" \
  -H "Content-Type: application/json" \
  -d '{
    "ml_output": {
      "predictions": [{
        "user_id": "USER_000000",
        "offer_flag": 1,
        "domain": "Dining",
        "confidence_score": 0.80
      }]
    },
    "use_llm_insights": false
  }'
```

### Python Example

```python
import requests

# Prepare ML predictions
payload = {
    "ml_output": {
        "predictions": [
            {
                "user_id": "USER_000000",
                "offer_flag": 1,
                "domain": "Electronics",
                "confidence_score": 0.9588919271748322
            }
        ]
    },
    "use_llm_insights": True
}

# Call API
response = requests.post(
    "http://localhost:9000/generate-offer",
    json=payload
)

# Get results
offers = response.json()
for offer in offers:
    if offer['success']:
        print(f"✅ Offer generated for {offer['user_id']}")
        print(f"   Title: {offer['offer']['offer_title']}")
        print(f"   Value: {offer['offer']['offer_value']}")
```

---

## 🔍 How It Works

### Step-by-Step Process

1. **ML Model Integration**
   - ML model predicts which users should receive offers
   - Outputs: user_id, offer_flag, domain, confidence_score
   - Predictions sent to GenAI system via API

2. **User Data Retrieval**
   - System fetches complete user profile from PostgreSQL
   - 40+ attributes including spending, engagement, and offer history
   - Flat column structure for efficient access

3. **Behavioral Analysis**
   - Rule-based engine calculates spending trends
   - Analyzes engagement patterns and offer history
   - LLM enhances with psychological insights (optional)

4. **Policy Retrieval**
   - RAG system queries ChromaDB for relevant policies
   - Retrieves domain-specific and compliance policies
   - Ensures offer will meet all requirements

5. **Offer Generation**
   - LLM creates personalized offer using:
     - User behavioral insights
     - ML recommendation
     - Retrieved policies
     - Business rules from config
   - Validates policy compliance
   - Generates compelling marketing copy

6. **Output & Export**
   - Returns JSON with offer details and reasoning
   - Generates HTML email template
   - Saves to output/ directory for records

---

## 📈 Performance

- **Rule-based analysis**: ~100ms per user
- **LLM-enhanced analysis**: ~2-3s per user (due to LLM calls)
- **Batch processing**: Parallel execution for multiple users
- **Database queries**: <50ms average (PostgreSQL with indexes)

**Recommendation**: Use `use_llm_insights: false` for real-time applications, `true` for deeper insights in batch processing.

---

## 🛠️ Troubleshooting

### Common Issues

**1. User not found**
```
Error: "User USER_XXX not found in database"
```
**Solution**: Ensure user_id exists in PostgreSQL `amex_data` table

**2. LLM timeout**
```
Error: "Error generating offer: timeout"
```
**Solution**: Set `use_llm_insights: false` or increase timeout in config

**3. No policies loaded**
```
Error: "No relevant policies found"
```
**Solution**: System auto-initializes default policies on startup. Check ChromaDB directory.

**4. Database connection failed**
```
Error: "Failed to connect to PostgreSQL"
```
**Solution**: Verify DB_CONFIG in user_data_store.py and network connectivity

---

## � Docker Instructions

### Build Docker Image

```cmd
docker build -t amex-offer-generator:latest .
```

### Run Locally

**With default hardcoded key**:
```cmd
docker run -d -p 9000:9000 --name amex-app amex-offer-generator:latest
```

**With custom OpenRouter API key** (recommended for production):
```cmd
docker run -d -p 9000:9000 -e OPENROUTER_API_KEY="your-api-key-here" --name amex-app amex-offer-generator:latest
```

The application will use the provided `OPENROUTER_API_KEY` environment variable if set, otherwise it falls back to the hardcoded key in the code.

### Publish to Docker Hub

**Step 1**: Create account at https://hub.docker.com

**Step 2**: Create repository named `amex-offer-generator`

**Step 3**: Build and push using the script:

```cmd
docker-publish.bat YOUR_DOCKERHUB_USERNAME
```

**Or manually**:

```cmd
# Build
docker build -t amex-offer-generator:latest .

# Tag
docker tag amex-offer-generator:latest YOUR_USERNAME/amex-offer-generator:latest

# Login to Docker Hub
docker login

# Push
docker push YOUR_USERNAME/amex-offer-generator:latest
```

### Pull and Run from Docker Hub

**With default key**:
```cmd
docker pull YOUR_USERNAME/amex-offer-generator:latest
docker run -d -p 9000:9000 --name amex-app YOUR_USERNAME/amex-offer-generator:latest
```

**With custom OpenRouter API key**:
```cmd
docker pull YOUR_USERNAME/amex-offer-generator:latest
docker run -d -p 9000:9000 -e OPENROUTER_API_KEY="your-api-key-here" --name amex-app YOUR_USERNAME/amex-offer-generator:latest
```

### Useful Commands

```cmd
# View logs
docker logs -f amex-app

# Stop container
docker stop amex-app

# Start container
docker start amex-app

# Remove container
docker rm amex-app

# Remove image
docker rmi amex-offer-generator:latest
```

---

## �📝 Notes

- User data is automatically fetched from PostgreSQL - no need to send full profiles in API requests
- System supports both flat and nested user data structures for backward compatibility
- Offers are saved to `output/` directory in JSON and HTML formats
- ChromaDB vector store persists in `chroma_db/` directory
- Set `OPENROUTER_API_KEY` and database credentials in `config.py` and `user_data_store.py` before running

---

## 📄 License

American Express Hackathon 2025 Project
