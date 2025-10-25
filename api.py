"""
FastAPI Server for GenAI Offer Generator

Provides REST API endpoints for generating personalized offers
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import os

from genai_reasoner import GenAIReasoner
from config import Config
from user_data_store import get_user_store


# Initialize FastAPI app
app = FastAPI(
    title="American Express - GenAI Offer Generator API",
    description="AI-powered personalized offer generation system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global reasoner instance (initialized on startup)
reasoner: Optional[GenAIReasoner] = None
user_store = None


# ==================== Request/Response Models ====================

class MLModelOutput(BaseModel):
    """ML Model Output - The only input required"""
    user_id: str = Field(..., description="Card Member identifier")
    offer_flag: bool = Field(..., description="Whether to trigger an offer now")
    domain: str = Field(..., description="Category: Travel, Dining, Retail, Entertainment")
    confidence_score: float = Field(..., ge=0, le=1, description="Probability of offer acceptance/lift")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "USR001",
                "offer_flag": True,
                "domain": "Travel",
                "confidence_score": 0.85
            }
        }


class GenerateOfferRequest(BaseModel):
    """Request with optional LLM insights flag"""
    ml_output: MLModelOutput
    use_llm_insights: bool = Field(
        default=True,
        description="Use LLM for enhanced behavioral analysis"
    )


class OfferResponse(BaseModel):
    success: bool
    user_id: str
    timestamp: str
    recommendation: str
    offer: Optional[Dict[str, Any]] = None
    behavioral_analysis: Optional[Dict[str, Any]] = None
    applied_policies: Optional[List[Dict[str, Any]]] = None
    email_html: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    config: Dict[str, Any]


class PolicySummaryResponse(BaseModel):
    total_policies: int
    policies: List[Dict[str, Any]]


# ==================== Startup/Shutdown Events ====================

@app.on_event("startup")
async def startup_event():
    """Initialize the GenAI Reasoner on server startup"""
    global reasoner, user_store
    
    print("=" * 80)
    print("Starting GenAI Offer Generator API Server...")
    print("=" * 80)
    
    try:
        # Validate configuration
        Config.validate()
        
        # Initialize user data store
        user_store = get_user_store()
        print(f"✅ User data store initialized with {user_store.get_user_count()} users")
        
        # Initialize the reasoner with default policies
        reasoner = GenAIReasoner(
            initialize_policies=True,
            use_llm_insights=True  # Default, can be overridden per request
        )
        
        print("✅ API Server initialized successfully")
        print(f"📚 Model: {Config.OPENROUTER_MODEL}")
        print(f"🌐 API Documentation: http://localhost:8000/docs")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Failed to initialize server: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown"""
    print("\n" + "=" * 80)
    print("Shutting down GenAI Offer Generator API Server...")
    print("=" * 80)


# ==================== API Endpoints ====================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint - API information"""
    return {
        "message": "American Express - GenAI Offer Generator API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    if reasoner is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return {
        "status": "healthy",
        "version": "1.0.0",
        "config": {
            "model": Config.OPENROUTER_MODEL,
            "embedding_model": Config.EMBEDDING_MODEL,
            "max_offers_per_month": Config.MAX_OFFERS_PER_MONTH,
            "min_confidence_score": Config.MIN_CONFIDENCE_SCORE,
            "domains": Config.DOMAINS
        }
    }


@app.get("/policies", response_model=PolicySummaryResponse)
async def get_policies():
    """Get all loaded policies"""
    
    if reasoner is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        policies = reasoner.get_policy_summary()
        
        return {
            "total_policies": len(policies),
            "policies": policies
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving policies: {str(e)}")


@app.post("/generate-offer", response_model=OfferResponse)
async def generate_offer(
    request: GenerateOfferRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate personalized offer for a user
    
    Accepts only ML model output:
    - user_id: Card member identifier
    - offer_flag: Whether to trigger an offer
    - domain: Category (Travel, Dining, Retail, Entertainment)
    - confidence_score: Probability of offer acceptance
    
    User data is automatically retrieved from the internal data store.
    """
    
    if reasoner is None or user_store is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        ml_output = request.ml_output.dict()
        user_id = ml_output['user_id']
        
        # Retrieve user data from store
        user_data = user_store.get_user_data(user_id)
        
        if user_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"User {user_id} not found in data store"
            )
        
        # Update reasoner's LLM insights setting if needed
        if request.use_llm_insights != reasoner.root_cause_analyzer.use_llm_insights:
            reasoner.root_cause_analyzer.use_llm_insights = request.use_llm_insights
        
        # Generate offer
        print(f"\n📥 Received request for user: {user_id}")
        print(f"   Domain: {ml_output['domain']}")
        print(f"   Confidence: {ml_output['confidence_score']:.1%}")
        print(f"   LLM Insights: {'Enabled' if request.use_llm_insights else 'Disabled'}")
        
        response = reasoner.process_ml_recommendation(user_data, ml_output)
        
        # Add success flag
        response['success'] = response.get('offer', {}).get('success', False)
        
        # Schedule background task to save offer (optional)
        if response['success']:
            background_tasks.add_task(
                save_offer_to_file,
                user_id,
                response
            )
        
        print(f"✅ Offer generated for {user_id}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error generating offer: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating offer: {str(e)}"
        )


@app.post("/add-policy")
async def add_custom_policy(
    policy_id: str,
    content: str,
    policy_type: str,
    domain: str = "all"
):
    """Add a custom policy to the system"""
    
    if reasoner is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        reasoner.add_custom_policy(policy_id, content, policy_type, domain)
        
        return {
            "success": True,
            "message": f"Policy '{policy_id}' added successfully",
            "policy_id": policy_id,
            "type": policy_type,
            "domain": domain
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding policy: {str(e)}")


# ==================== Batch Processing Endpoint ====================

@app.post("/batch-generate-offers")
async def batch_generate_offers(
    requests: List[GenerateOfferRequest],
    background_tasks: BackgroundTasks
):
    """
    Generate offers for multiple users in batch
    
    Each request contains only ML model output.
    User data is retrieved automatically for each user_id.
    """
    
    if reasoner is None or user_store is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if len(requests) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size limited to 100 users. Use multiple requests for larger batches."
        )
    
    results = []
    
    for idx, request in enumerate(requests, 1):
        try:
            ml_output = request.ml_output.dict()
            user_id = ml_output['user_id']
            
            print(f"\nProcessing user {idx}/{len(requests)}: {user_id}")
            
            # Retrieve user data
            user_data = user_store.get_user_data(user_id)
            
            if user_data is None:
                results.append({
                    "success": False,
                    "user_id": user_id,
                    "error": "User not found in data store"
                })
                continue
            
            response = reasoner.process_ml_recommendation(user_data, ml_output)
            response['success'] = response.get('offer', {}).get('success', False)
            
            results.append(response)
            
        except Exception as e:
            print(f"❌ Error processing user {request.ml_output.user_id}: {e}")
            results.append({
                "success": False,
                "user_id": request.ml_output.user_id,
                "error": str(e)
            })
    
    return {
        "total_requests": len(requests),
        "successful": sum(1 for r in results if r.get('success')),
        "failed": sum(1 for r in results if not r.get('success')),
        "results": results
    }


# ==================== Helper Functions ====================

def save_offer_to_file(user_id: str, response: Dict[str, Any]):
    """Background task to save generated offer to file"""
    try:
        os.makedirs('./output/api', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./output/api/offer_{user_id}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(response, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved offer to {filename}")
        
    except Exception as e:
        print(f"⚠️  Failed to save offer: {e}")


# ==================== Run Server ====================

if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║              American Express - GenAI Offer Generator API                   ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Set to False in production
        log_level="info"
    )
