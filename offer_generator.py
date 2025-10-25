"""
Offer Generator - Uses OpenRouter LLM to generate personalized, policy-compliant offers
"""

import json
from typing import Dict, List, Any
from openai import OpenAI
from config import Config


class OfferGenerator:
    """Generates personalized offers using LLM with policy constraints"""
    
    def __init__(self):
        """Initialize the offer generator with OpenRouter API using ChatOpenAI"""
        self.api_key = Config.OPENROUTER_API_KEY
        self.model = Config.OPENROUTER_MODEL
        
        if not self.api_key:
            raise ValueError("OpenRouter API key not configured. Please set OPENROUTER_API_KEY in config.py")
        
        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=Config.OPENROUTER_BASE_URL
        )
    
    def generate_offer(
        self,
        user_data: Dict[str, Any],
        behavioral_analysis: Dict[str, Any],
        relevant_policies: List[Dict[str, Any]],
        ml_recommendation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a personalized offer using LLM
        
        Args:
            user_data: User profile and transaction data
            behavioral_analysis: Output from RootCauseAnalyzer
            relevant_policies: Retrieved policies from RAG system
            ml_recommendation: ML model output
            
        Returns:
            Dict containing the generated offer and metadata
        """
        
        # Build the prompt for the LLM
        prompt = self._build_offer_prompt(
            user_data,
            behavioral_analysis,
            relevant_policies,
            ml_recommendation
        )
        
        # Call OpenRouter API
        try:
            response = self._call_openrouter(prompt)
            
            # Parse the response
            offer_data = self._parse_llm_response(response, ml_recommendation)
            
            return offer_data
            
        except Exception as e:
            print(f"Error generating offer: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _build_offer_prompt(
        self,
        user_data: Dict[str, Any],
        behavioral_analysis: Dict[str, Any],
        relevant_policies: List[Dict[str, Any]],
        ml_recommendation: Dict[str, Any]
    ) -> str:
        """Build the prompt for the LLM"""
        
        # Extract user information
        user_id = user_data.get('user_id', 'Unknown')
        user_name = user_data.get('profile', {}).get('name', 'Valued Card Member')
        segment = user_data.get('profile', {}).get('segment', 'Standard')
        tenure = user_data.get('profile', {}).get('tenure_months', 0)
        
        # Extract behavioral insights
        insights = behavioral_analysis.get('insights', [])
        domain = ml_recommendation.get('domain', 'General')
        confidence = ml_recommendation.get('confidence_score', 0)
        
        spending_trends = behavioral_analysis.get('spending_trends', {})
        engagement = behavioral_analysis.get('engagement_metrics', {})
        
        # Format policies
        policy_context = "\n\n".join([
            f"Policy: {p['metadata'].get('type', 'general')}\n{p['content']}"
            for p in relevant_policies
        ])
        
        # Build comprehensive prompt
        prompt = f"""You are an expert offer designer for American Express. Your task is to create a highly personalized, compelling offer for a card member while strictly adhering to company policies.

**USER PROFILE:**
- User ID: {user_id}
- Name: {user_name}
- Segment: {segment}
- Account Tenure: {tenure} months

**BEHAVIORAL ANALYSIS:**
ML Recommendation: Target {domain} category with {confidence:.0%} confidence

Key Insights:
{chr(10).join(f"- {insight}" for insight in insights)}

Spending Trends in {domain}:
- Total Spend: ${spending_trends.get('total_spend', 0):,.2f}
- Transaction Count: {spending_trends.get('transaction_count', 0)}
- Trend: {spending_trends.get('trend', 'unknown')} ({spending_trends.get('trend_percentage', 0):+.1f}%)

Engagement Level: {engagement.get('engagement_level', 'unknown')}
- Offer Click Rate: {engagement.get('offer_click_rate', 0):.1%}
- Redemption Rate: {engagement.get('redemption_rate', 0):.1%}

**APPLICABLE POLICIES AND CONSTRAINTS:**
{policy_context}

**YOUR TASK:**
Generate a personalized offer that:
1. Addresses the behavioral patterns identified in the analysis
2. Is compelling and relevant to the {domain} category
3. Strictly complies with ALL policies listed above
4. Uses appropriate American Express tone: premium, exclusive, member-focused
5. Includes clear terms and a call-to-action
6. Provides reasoning for why this offer is suitable for this specific user

**OUTPUT FORMAT (respond ONLY with valid JSON):**
{{
    "offer_title": "Brief, compelling offer headline (max 60 chars)",
    "offer_description": "Full offer description with benefits and value proposition (2-3 sentences)",
    "terms_and_conditions": "Clear, specific terms (validity period, spending requirements, exclusions)",
    "call_to_action": "Specific action user should take",
    "offer_type": "points_multiplier|statement_credit|cashback|discount|exclusive_access",
    "offer_value": "Quantified benefit (e.g., '3X Points', '$50 Credit', '20% Back')",
    "expiration_days": 30-60 days (integer),
    "minimum_spend": Minimum spend requirement in dollars (integer),
    "category_restrictions": ["list", "of", "specific", "eligible", "categories"],
    "reasoning": "2-3 sentence explanation of why this offer is optimal for this user based on their behavior",
    "policy_compliance_notes": ["list of policies verified and how offer complies"]
}}

Generate the offer now:"""
        
        return prompt
    
    def _call_openrouter(self, prompt: str) -> str:
        """Call OpenRouter API using ChatOpenAI interface"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert offer designer for American Express. You create personalized, policy-compliant offers. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2000,
                top_p=0.9,
                extra_headers={
                    "HTTP-Referer": "https://github.com/amex-hackathon",
                    "X-Title": "AMEX Offer Generator"
                }
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error calling OpenRouter API: {str(e)}")
            raise
    
    def _parse_llm_response(
        self, 
        response: str, 
        ml_recommendation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse and validate LLM response"""
        
        try:
            # Try to extract JSON from response (LLM might add extra text)
            # Look for JSON content between curly braces
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                offer_data = json.loads(json_str)
            else:
                offer_data = json.loads(response)
            
            # Add metadata
            offer_data['success'] = True
            offer_data['ml_confidence'] = ml_recommendation.get('confidence_score', 0)
            offer_data['domain'] = ml_recommendation.get('domain')
            offer_data['generated_at'] = pd.Timestamp.now().isoformat()
            
            # Validate required fields
            required_fields = [
                'offer_title', 'offer_description', 'call_to_action', 
                'offer_type', 'offer_value'
            ]
            
            for field in required_fields:
                if field not in offer_data:
                    raise ValueError(f"Missing required field: {field}")
            
            return offer_data
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {str(e)}")
            print(f"Raw response: {response}")
            
            # Return a fallback structure
            return {
                'success': False,
                'error': 'Failed to parse LLM response',
                'raw_response': response,
                'offer_title': 'Offer Generation Failed',
                'offer_description': 'Unable to generate offer at this time.'
            }
    
    def format_offer_for_email(self, offer_data: Dict[str, Any]) -> str:
        """Format the offer as an email-ready HTML template"""
        
        if not offer_data.get('success', False):
            return "<p>Unable to generate offer at this time.</p>"
        
        html_template = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background-color: #f5f5f5;">
            <div style="background-color: #006FCF; color: white; padding: 20px; text-align: center;">
                <h1 style="margin: 0; font-size: 24px;">Exclusive Offer Just for You</h1>
            </div>
            
            <div style="background-color: white; padding: 30px; margin-top: 2px;">
                <h2 style="color: #006FCF; margin-top: 0;">{offer_data.get('offer_title', '')}</h2>
                
                <div style="background-color: #f0f7ff; padding: 15px; border-left: 4px solid #006FCF; margin: 20px 0;">
                    <p style="font-size: 18px; font-weight: bold; margin: 0; color: #006FCF;">
                        {offer_data.get('offer_value', '')}
                    </p>
                </div>
                
                <p style="font-size: 16px; line-height: 1.6; color: #333;">
                    {offer_data.get('offer_description', '')}
                </p>
                
                <div style="text-align: center; margin: 30px 0;">
                    <a href="#activate" style="background-color: #006FCF; color: white; padding: 15px 40px; text-decoration: none; border-radius: 4px; font-weight: bold; display: inline-block;">
                        {offer_data.get('call_to_action', 'Activate Offer')}
                    </a>
                </div>
                
                <div style="border-top: 1px solid #ddd; padding-top: 20px; margin-top: 20px;">
                    <h3 style="font-size: 14px; color: #666; margin-bottom: 10px;">Terms & Conditions:</h3>
                    <p style="font-size: 12px; color: #666; line-height: 1.5;">
                        {offer_data.get('terms_and_conditions', '')}
                    </p>
                    {f'<p style="font-size: 12px; color: #666;">Minimum spend: ${offer_data.get("minimum_spend", 0)}</p>' if offer_data.get('minimum_spend') else ''}
                    {f'<p style="font-size: 12px; color: #666;">Valid for: {offer_data.get("expiration_days", 30)} days</p>' if offer_data.get('expiration_days') else ''}
                </div>
            </div>
            
            <div style="text-align: center; padding: 20px; font-size: 12px; color: #999;">
                <p>American Express® | Member Since [Year]</p>
                <p>This offer was personalized based on your spending preferences</p>
            </div>
        </div>
        """
        
        return html_template


# Import pandas for timestamp
import pandas as pd
