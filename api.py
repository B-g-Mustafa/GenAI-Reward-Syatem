"""
FastAPI Server for GenAI Offer Generator

Provides REST API endpoints for generating personalized offers
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
from decimal import Decimal
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

class MLPrediction(BaseModel):
    """Single ML Model Prediction"""
    user_id: str = Field(..., description="Card Member identifier")
    offer_flag: int = Field(..., description="Whether to trigger an offer (0 or 1)")
    domain: str = Field(..., description="Category: Travel, Business Services, Retail, Electronics, Dining, Entertainment")
    confidence_score: float = Field(..., ge=0, le=1, description="Probability of offer acceptance/lift")
    pred_version: Optional[str] = Field(None, description="Model version")
    prediction_timestamp: Optional[str] = Field(None, description="When prediction was made")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "USER_000000",
                "offer_flag": 1,
                "domain": "Electronics",
                "confidence_score": 0.9588919271748322,
                "pred_version": "v1",
                "prediction_timestamp": "2025-10-28T10:28:16.506594"
            }
        }


class MLModelOutput(BaseModel):
    """ML Model Batch Output - Accepts array of predictions"""
    total_users: Optional[int] = Field(None, description="Total number of users processed")
    predictions: List[MLPrediction] = Field(..., description="Array of user predictions")
    processing_time_seconds: Optional[float] = Field(None, description="Time taken for ML processing")
    
    class Config:
        schema_extra = {
            "example": {
                "total_users": 1,
                "predictions": [
                    {
                        "user_id": "USER_000000",
                        "offer_flag": 1,
                        "domain": "Electronics",
                        "confidence_score": 0.9588919271748322,
                        "pred_version": "v1",
                        "prediction_timestamp": "2025-10-28T10:28:16.506594"
                    }
                ],
                "processing_time_seconds": 0.037884
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
        print(f"🌐 API Documentation: http://localhost:9000/docs")
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


@app.post("/generate-offer", response_model=List[OfferResponse])
async def generate_offer(
    request: GenerateOfferRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate personalized offers for users from ML predictions
    
    Accepts ML model output with predictions array:
    - Each prediction contains: user_id, offer_flag, domain, confidence_score
    - User data is automatically retrieved from PostgreSQL database
    - Only processes users with offer_flag=1
    """
    
    if reasoner is None or user_store is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        predictions = request.ml_output.predictions
        results = []
        
        print(f"\n📥 Received {len(predictions)} prediction(s)")
        
        # Update reasoner's LLM insights setting if needed
        if request.use_llm_insights != reasoner.root_cause_analyzer.use_llm_insights:
            reasoner.root_cause_analyzer.use_llm_insights = request.use_llm_insights
        
        # Process each prediction
        for idx, prediction in enumerate(predictions, 1):
            pred_dict = prediction.dict()
            user_id = pred_dict['user_id']
            offer_flag = pred_dict['offer_flag']
            
            print(f"\n[{idx}/{len(predictions)}] Processing user: {user_id}")
            
            # Skip if offer_flag is 0 (no offer needed)
            if offer_flag == 0:
                print(f"   ⏭️  Skipping - offer_flag=0")
                results.append({
                    "success": False,
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "recommendation": "no_offer_needed",
                    "error": "ML model indicated no offer needed (offer_flag=0)"
                })
                continue
            
            # Retrieve user data from PostgreSQL
            user_data = user_store.get_user_data(user_id)
            
            if user_data is None:
                print(f"   ❌ User not found in database")
                results.append({
                    "success": False,
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "recommendation": "user_not_found",
                    "error": f"User {user_id} not found in database"
                })
                continue
            
            # Generate offer
            print(f"   Domain: {pred_dict['domain']}")
            print(f"   Confidence: {pred_dict['confidence_score']:.1%}")
            print(f"   LLM Insights: {'Enabled' if request.use_llm_insights else 'Disabled'}")
            
            try:
                response = reasoner.process_ml_recommendation(user_data, pred_dict)
                
                # Add success flag
                response['success'] = response.get('offer', {}).get('success', False)
                
                # Schedule background task to save offer (optional)
                if response['success']:
                    background_tasks.add_task(
                        save_offer_to_file,
                        user_id,
                        response
                    )
                
                results.append(response)
                print(f"   ✅ Offer generated successfully")
                
            except Exception as e:
                print(f"   ❌ Error generating offer: {e}")
                results.append({
                    "success": False,
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "recommendation": "error",
                    "error": str(e)
                })
        
        print(f"\n✅ Processed {len(results)} users")
        print(f"   Successful: {sum(1 for r in results if r.get('success'))}")
        print(f"   Failed: {sum(1 for r in results if not r.get('success'))}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error processing predictions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing predictions: {str(e)}"
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
    Generate offers for multiple batches of ML predictions
    
    Each request contains ML output with predictions array.
    User data is retrieved automatically from PostgreSQL for each user_id.
    """
    
    if reasoner is None or user_store is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    all_results = []
    total_predictions = sum(len(req.ml_output.predictions) for req in requests)
    
    if total_predictions > 500:
        raise HTTPException(
            status_code=400,
            detail="Total predictions limited to 500. Use multiple requests for larger batches."
        )
    
    print(f"\n📥 Processing {len(requests)} batch(es) with {total_predictions} total predictions")
    
    for batch_idx, request in enumerate(requests, 1):
        predictions = request.ml_output.predictions
        print(f"\n=== Batch {batch_idx}/{len(requests)}: {len(predictions)} predictions ===")
        
        for idx, prediction in enumerate(predictions, 1):
            try:
                pred_dict = prediction.dict()
                user_id = pred_dict['user_id']
                offer_flag = pred_dict['offer_flag']
                
                print(f"[{idx}/{len(predictions)}] Processing user: {user_id}")
                
                # Skip if offer_flag is 0
                if offer_flag == 0:
                    print(f"   ⏭️  Skipping - offer_flag=0")
                    all_results.append({
                        "success": False,
                        "user_id": user_id,
                        "timestamp": datetime.now().isoformat(),
                        "recommendation": "no_offer_needed",
                        "error": "ML model indicated no offer needed"
                    })
                    continue
                
                # Retrieve user data from PostgreSQL
                user_data = user_store.get_user_data(user_id)
                
                if user_data is None:
                    print(f"   ❌ User not found")
                    all_results.append({
                        "success": False,
                        "user_id": user_id,
                        "timestamp": datetime.now().isoformat(),
                        "recommendation": "user_not_found",
                        "error": "User not found in database"
                    })
                    continue
                
                # Generate offer
                response = reasoner.process_ml_recommendation(user_data, pred_dict)
                response['success'] = response.get('offer', {}).get('success', False)
                
                if response['success']:
                    background_tasks.add_task(save_offer_to_file, user_id, response)
                
                all_results.append(response)
                print(f"   ✅ Success")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                all_results.append({
                    "success": False,
                    "user_id": prediction.user_id,
                    "timestamp": datetime.now().isoformat(),
                    "recommendation": "error",
                    "error": str(e)
                })
    
    successful = sum(1 for r in all_results if r.get('success'))
    failed = sum(1 for r in all_results if not r.get('success'))
    
    print(f"\n✅ Batch processing complete:")
    print(f"   Total: {len(all_results)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    
    return {
        "total_batches": len(requests),
        "total_predictions": len(all_results),
        "successful": successful,
        "failed": failed,
        "results": all_results
    }


# ==================== Helper Functions ====================

class DecimalEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Decimal types from PostgreSQL"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            # Convert Decimal to float for JSON serialization
            return float(obj)
        return super(DecimalEncoder, self).default(obj)


def save_offer_to_file(user_id: str, response: Dict[str, Any]):
    """Background task to save generated offer to file"""
    try:
        os.makedirs('./output/api', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./output/api/offer_{user_id}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(response, f, indent=2, ensure_ascii=False, cls=DecimalEncoder)
        
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
        port=9000,
        reload=True,  # Set to False in production
        log_level="info"
    )
