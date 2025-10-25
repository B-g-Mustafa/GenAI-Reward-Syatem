"""
GenAI Reasoner - Main orchestrator that coordinates all components
"""

from typing import Dict, Any, List
import json
from datetime import datetime

from config import Config
from rag_system import PolicyRAG
from root_cause_analyzer import RootCauseAnalyzer
from offer_generator import OfferGenerator


class GenAIReasoner:
    """
    Main GenAI Reasoner and Offer Generator System
    
    Coordinates:
    1. Policy retrieval via RAG
    2. Behavioral analysis and root-cause reasoning
    3. Personalized offer generation via LLM
    """
    
    def __init__(self, initialize_policies: bool = True, use_llm_insights: bool = True):
        """
        Initialize the GenAI Reasoner system
        
        Args:
            initialize_policies: Whether to initialize default policies
            use_llm_insights: Whether to use LLM for enhanced behavioral analysis (default: True)
        """
        # Validate configuration
        Config.validate()
        
        # Initialize components
        print("Initializing GenAI Reasoner System...")
        
        self.policy_rag = PolicyRAG()
        self.root_cause_analyzer = RootCauseAnalyzer(use_llm_insights=use_llm_insights)
        self.offer_generator = OfferGenerator()
        
        # Initialize policies if requested
        if initialize_policies:
            print("Loading default policies...")
            self.policy_rag.initialize_default_policies()
        
        mode = "with LLM-enhanced insights" if use_llm_insights else "rule-based only"
        print(f"✓ GenAI Reasoner System ready ({mode})")

    
    def process_ml_recommendation(
        self,
        user_data: Dict[str, Any],
        ml_recommendation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main pipeline: Process ML recommendation and generate personalized offer
        
        Args:
            user_data: Complete user profile and transaction data
            ml_recommendation: Output from ML model containing:
                - user_id: Card member ID
                - offer_flag: Boolean indicating if offer should be made
                - domain: Category (Travel, Dining, etc.)
                - confidence_score: ML confidence (0-1)
        
        Returns:
            Complete offer package with analysis and generated offer
        """
        
        print(f"\n{'='*60}")
        print(f"Processing recommendation for User: {user_data.get('user_id')}")
        print(f"Domain: {ml_recommendation.get('domain')} | "
              f"Confidence: {ml_recommendation.get('confidence_score', 0):.1%}")
        print(f"{'='*60}\n")
        
        # Check if we should make an offer
        if not ml_recommendation.get('offer_flag', False):
            print("❌ ML model suggests no offer at this time")
            return {
                'user_id': user_data.get('user_id'),
                'recommendation': 'no_offer',
                'reason': 'ML model confidence too low or timing not optimal',
                'ml_output': ml_recommendation
            }
        
        # Step 1: Behavioral Analysis and Root-Cause Reasoning
        print("📊 Step 1: Analyzing user behavior...")
        behavioral_analysis = self.root_cause_analyzer.analyze_user_behavior(
            user_data,
            ml_recommendation
        )
        
        print(f"  ✓ Generated {len(behavioral_analysis.get('insights', []))} insights")
        
        # Step 2: Retrieve Relevant Policies via RAG
        print("\n📚 Step 2: Retrieving relevant policies...")
        
        # Build query for policy retrieval
        domain = ml_recommendation.get('domain')
        query = self._build_policy_query(user_data, behavioral_analysis, domain)
        
        relevant_policies = self.policy_rag.retrieve_relevant_policies(
            query=query,
            domain=domain,
            n_results=5
        )
        
        # Always include compliance policies
        compliance_policies = self.policy_rag.retrieve_relevant_policies(
            query="compliance regulatory requirements",
            n_results=2
        )
        
        all_policies = relevant_policies + compliance_policies
        print(f"  ✓ Retrieved {len(all_policies)} relevant policies")
        
        # Step 3: Generate Personalized Offer using LLM
        print("\n🤖 Step 3: Generating personalized offer with LLM...")
        
        offer_data = self.offer_generator.generate_offer(
            user_data=user_data,
            behavioral_analysis=behavioral_analysis,
            relevant_policies=all_policies,
            ml_recommendation=ml_recommendation
        )
        
        if offer_data.get('success'):
            print(f"  ✓ Offer generated: {offer_data.get('offer_title')}")
            print(f"  ✓ Offer value: {offer_data.get('offer_value')}")
        else:
            print(f"  ❌ Offer generation failed: {offer_data.get('error')}")
        
        # Step 4: Compile complete response
        print("\n📦 Step 4: Compiling complete offer package...")
        
        response = {
            'user_id': user_data.get('user_id'),
            'timestamp': datetime.now().isoformat(),
            'recommendation': 'offer_generated' if offer_data.get('success') else 'generation_failed',
            
            # ML Recommendation
            'ml_recommendation': ml_recommendation,
            
            # Behavioral Analysis
            'behavioral_analysis': {
                'insights': behavioral_analysis.get('insights', []),
                'spending_trends': behavioral_analysis.get('spending_trends', {}),
                'engagement_metrics': behavioral_analysis.get('engagement_metrics', {}),
                'offer_performance': behavioral_analysis.get('offer_performance', {})
            },
            
            # Retrieved Policies
            'applied_policies': [
                {
                    'id': p['id'],
                    'type': p['metadata'].get('type'),
                    'domain': p['metadata'].get('domain')
                }
                for p in all_policies
            ],
            
            # Generated Offer
            'offer': offer_data,
            
            # Email-ready format
            'email_html': self.offer_generator.format_offer_for_email(offer_data) if offer_data.get('success') else None
        }
        
        print("\n✅ Processing complete!")
        print(f"{'='*60}\n")
        
        return response
    
    def _build_policy_query(
        self, 
        user_data: Dict[str, Any], 
        behavioral_analysis: Dict[str, Any],
        domain: str
    ) -> str:
        """Build query for RAG policy retrieval"""
        
        segment = user_data.get('profile', {}).get('segment', 'standard')
        spending_trend = behavioral_analysis.get('spending_trends', {}).get('trend', 'stable')
        
        query = f"""
        {domain} category offer for {segment} segment card member.
        Spending trend: {spending_trend}.
        Need offer terms, eligibility, and merchant requirements.
        """
        
        return query.strip()
    
    def get_policy_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all loaded policies"""
        policies = self.policy_rag.get_all_policies()
        
        summary = []
        for policy in policies:
            summary.append({
                'id': policy['id'],
                'type': policy['metadata'].get('type', 'unknown'),
                'domain': policy['metadata'].get('domain', 'all'),
                'preview': policy['content'][:100] + '...'
            })
        
        return summary
    
    def add_custom_policy(
        self, 
        policy_id: str, 
        content: str, 
        policy_type: str,
        domain: str = 'all'
    ):
        """Add a custom policy to the system"""
        
        metadata = {
            'type': policy_type,
            'domain': domain,
            'custom': True
        }
        
        self.policy_rag.add_policy(policy_id, content, metadata)
        print(f"✓ Added custom policy: {policy_id}")
    
    def export_offer(self, offer_response: Dict[str, Any], filepath: str):
        """Export offer response to JSON file"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(offer_response, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Offer exported to: {filepath}")
    
    def export_offer_email(self, offer_response: Dict[str, Any], filepath: str):
        """Export email HTML to file"""
        
        html = offer_response.get('email_html')
        if not html:
            print("❌ No email HTML available in offer response")
            return
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✓ Email HTML exported to: {filepath}")
