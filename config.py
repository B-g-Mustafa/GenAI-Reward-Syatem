import os

class Config:
    """Configuration class for GenAI Offer Generator"""
    
    # OpenRouter API Configuration
    # Uses environment variable if provided, otherwise falls back to hardcoded key
    OPENROUTER_API_KEY = os.environ.get(
        "OPENROUTER_API_KEY", 
        "YOUR_API_KEY"
    )
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
    
    # Embedding and Vector DB Configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    VECTOR_DB_PATH = "./chroma_db"
    
    # Offer Generation Parameters
    MAX_OFFERS_PER_MONTH = 3
    MIN_CONFIDENCE_SCORE = 0.6
    
    # Domain Categories
    DOMAINS = ["Travel", "Business Services", "Retail", "Electronics", "Dining", "Entertainment"]
    
    @staticmethod
    def validate():
        """Validate required configuration"""
        if not Config.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not set. Please configure it in config.py")
        return True
