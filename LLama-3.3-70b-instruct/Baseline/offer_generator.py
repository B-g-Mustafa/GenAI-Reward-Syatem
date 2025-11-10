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
        
        # Extract user information (flat structure)
        user_id = user_data.get('user_id', 'Unknown')
        user_name = user_data.get('name', 'Valued Card Member')
        segment = user_data.get('segment', 'Standard')
        tenure_years = user_data.get('tenure_years', 0)
        age = user_data.get('age', 'N/A')
        gender = user_data.get('gender', 'N/A')
        location = user_data.get('location', 'N/A')
        persona = user_data.get('persona', 'N/A')
        card_type = user_data.get('card_type', 'Standard')
        customer_lifecycle_stage = user_data.get('customer_lifecycle_stage', 'N/A')
        churn_risk_score = user_data.get('churn_risk_score', 0)
        
        # Extract spending and engagement metrics
        total_transactions_12m = user_data.get('total_transactions_12m', 0)
        total_spend_12m = user_data.get('total_spend_12m', 0)
        avg_transaction_amount = user_data.get('avg_transaction_amount', 0)
        recency_days = user_data.get('recency_days', 0)
        frequency_30d = user_data.get('frequency_30d', 0)
        monetary_30d = user_data.get('monetary_30d', 0)
        
        # Engagement metrics
        email_open_rate = user_data.get('email_open_rate', 0)
        total_app_opens_90d = user_data.get('total_app_opens_90d', 0)
        offer_views_90d = user_data.get('offer_views_90d', 0)
        offer_clicks_90d = user_data.get('offer_clicks_90d', 0)
        
        # Offer history
        offers_shown_6m = user_data.get('offers_shown_6m', 0)
        offers_accepted_6m = user_data.get('offers_accepted_6m', 0)
        historical_acceptance_rate = user_data.get('historical_acceptance_rate', 0)
        days_since_last_offer = user_data.get('days_since_last_offer', 999)
        last_offer_domain = user_data.get('last_offer_domain', 'N/A')
        
        # Extract behavioral insights
        insights = behavioral_analysis.get('insights', [])
        domain = ml_recommendation.get('domain', 'General')
        confidence = ml_recommendation.get('confidence_score', 0)
        
        spending_trends = behavioral_analysis.get('spending_trends', {})
        engagement = behavioral_analysis.get('engagement_metrics', {})
        
        # Separate policies and merchant data
        policies = []
        merchant_info = ""
        
        for p in relevant_policies:
            if p['metadata'].get('type') == 'merchants':
                merchant_info = p['content']
            else:
                policies.append(p)
        
        # Format policies
        policy_context = "\n\n".join([
            f"Policy: {p['metadata'].get('type', 'general')}\n{p['content']}"
            for p in policies
        ])
        
        # Build comprehensive prompt
        prompt = f"""You are an expert offer designer for American Express. Your task is to create a highly personalized, compelling offer for a card member while strictly adhering to company policies.

**USER PROFILE:**
- User ID: {user_id}
- Name: {user_name}
- Age: {age}
- Gender: {gender}
- Location: {location}
- Segment: {segment}
- Card Type: {card_type}
- Account Tenure: {tenure_years} years
- Persona: {persona}
- Lifecycle Stage: {customer_lifecycle_stage}
- Churn Risk: {churn_risk_score}

**SPENDING PATTERNS (Last 12 Months):**
- Total Transactions: {total_transactions_12m}
- Total Spend: ${total_spend_12m:,.2f}
- Avg Transaction: ${avg_transaction_amount:.2f}
- Recency: {recency_days} days since last transaction
- Recent Activity: {frequency_30d} transactions in last 30 days (${monetary_30d:,.2f})

**ENGAGEMENT METRICS:**
- Email Open Rate: {email_open_rate:.1%}
- App Opens (90d): {total_app_opens_90d}
- Offer Views (90d): {offer_views_90d}
- Offer Clicks (90d): {offer_clicks_90d}

**OFFER HISTORY:**
- Offers Shown (6m): {offers_shown_6m}
- Offers Accepted (6m): {offers_accepted_6m}
- Historical Acceptance Rate: {historical_acceptance_rate:.1%}
- Days Since Last Offer: {days_since_last_offer}
- Last Offer Domain: {last_offer_domain}

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

**PARTNER MERCHANTS FOR {domain.upper()}:**
{merchant_info if merchant_info else "General merchant network available"}

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
    "featured_merchants": [
        {{
            "merchant_name": "Merchant Name",
            "merchant_category": "Specific category",
            "merchant_id": "MERCH_ID (from partner list above)",
            "merchant_type": "online|in-store|both"
        }}
    ],
    "reasoning": "2-3 sentence explanation of why this offer is optimal for this user based on their behavior",
    "policy_compliance_notes": ["list of policies verified and how offer complies"]
}}

**CRITICAL MERCHANT SELECTION RULES:**
1. Select ONLY 2-4 merchants (not all merchants from the category)
2. Choose the most recognizable and popular merchants that fit the user's profile
3. Include merchant names in the offer_description and call_to_action
4. Ensure merchant_id matches exactly from the partner list above
5. Do NOT list all merchants - be selective and strategic

Example good selection for Electronics: ["Best Buy", "Apple Store"] ✓
Example bad selection: Listing all 15+ electronics merchants ✗

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
            
            # Validate and limit merchant count (should be 2-4 merchants only)
            featured_merchants = offer_data.get('featured_merchants', [])
            if len(featured_merchants) > 4:
                print(f"Warning: LLM included {len(featured_merchants)} merchants, truncating to 4")
                print(f"Full list: {[m.get('merchant_name') for m in featured_merchants]}")
                offer_data['featured_merchants'] = featured_merchants[:4]
                offer_data['validation_warning'] = f"Merchant list truncated from {len(featured_merchants)} to 4"
            elif len(featured_merchants) < 2:
                print(f"Warning: Only {len(featured_merchants)} merchants included, expected 2-4")
                if len(featured_merchants) == 0:
                    offer_data['validation_warning'] = "No merchants included in offer"
            
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
