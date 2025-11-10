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
        """Build the prompt for the LLM with causal archetype strategy"""
        
        # Extract user information (flat structure)
        user_id = user_data.get('user_id', 'Unknown')
        user_name = user_data.get('name', 'Valued Card Member')
        segment = user_data.get('segment', 'Standard')
        tenure_years = user_data.get('tenure_years', 0)
        churn_risk_score = user_data.get('churn_risk_score', 0)
        
        # Extract spending and engagement metrics
        total_spend_12m = user_data.get('total_spend_12m', 0)
        historical_acceptance_rate = user_data.get('historical_acceptance_rate', 0)
        
        # Extract causal analysis
        causal_analysis = behavioral_analysis.get('causal_analysis', {})
        user_archetype = causal_analysis.get('user_archetype', 'Unknown')
        archetype_justification = causal_analysis.get('justification', '')
        expected_uplift = causal_analysis.get('expected_uplift', 0)
        counterfactual_scenario = causal_analysis.get('counterfactual_scenario', '')
        
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
        prompt = f"""You are an expert offer strategist for American Express specializing in causal targeting and personalized incentive design. Your task is to create a highly personalized offer that maximizes incremental impact while adhering to company policies.

    **CAUSAL ARCHETYPE CLASSIFICATION:**
    This user has been classified as: **{user_archetype}**

    Expected Uplift: {expected_uplift:+.1%}
    Justification: {archetype_justification}

    Counterfactual Baseline: {counterfactual_scenario}

    **STRATEGIC GUIDANCE BY ARCHETYPE:**

    If **PERSUADABLE** (High Uplift Potential):
    - STRATEGY: Maximize conversion with compelling, high-value offers
    - APPROACH: Overcome inertia with strong incentive that tips the decision
    - VALUE: Medium to high offer value (statement credits, bonus points)
    - MESSAGING: Emphasize urgency, exclusivity, and tangible benefits
    - GOAL: Convert fence-sitters who won't act without meaningful incentive
    - JUSTIFICATION: Focus on how offer overcomes their specific barriers

    If **SURE THING** (Will Convert Anyway):
    - STRATEGY: Minimize cost while maintaining satisfaction
    - APPROACH: Low-cost loyalty reinforcement or brand appreciation
    - VALUE: Low to modest offer value (small bonus, recognition)
    - MESSAGING: Emphasize appreciation, relationship, premium status
    - GOAL: Reward loyalty without overspending on guaranteed conversions
    - JUSTIFICATION: Focus on relationship maintenance, not behavioral change

    If **LOST CAUSE** (Unlikely to Respond):
    - STRATEGY: Minimal investment or diagnostic offer
    - APPROACH: Test engagement with low-cost, low-risk offer
    - VALUE: Very low cost (targeted micro-incentive)
    - MESSAGING: Re-engagement focused, testing receptiveness
    - GOAL: Avoid resource waste; probe for any response signal
    - JUSTIFICATION: Acknowledge low probability but test for hidden potential

    **USER PROFILE:**
    - User ID: {user_id}
    - Name: {user_name}
    - Segment: {segment}
    - Account Tenure: {tenure_years} years
    - Churn Risk: {churn_risk_score}
    - Total Spend (12m): ${total_spend_12m:,.2f}
    - Historical Offer Acceptance: {historical_acceptance_rate:.1%}

    **ML RECOMMENDATION:**
    - Target Category: {domain}
    - Confidence: {confidence:.0%}

    **BEHAVIORAL INSIGHTS:**
    {chr(10).join(f"- {insight}" for insight in insights)}

    Spending Trends in {domain}:
    - Total Spend: ${spending_trends.get('total_spend', 0):,.2f}
    - Trend: {spending_trends.get('trend', 'unknown')} ({spending_trends.get('trend_percentage', 0):+.1f}%)

    Engagement Level: {engagement.get('engagement_level', 'unknown')}
    - Offer Click Rate: {engagement.get('offer_click_rate', 0):.1%}

    **APPLICABLE POLICIES:**
    {policy_context}

    **PARTNER MERCHANTS FOR {domain.upper()}:**
    {merchant_info if merchant_info else "General merchant network available"}

    **YOUR OFFER DESIGN TASK:**

    1. **Apply Archetype-Specific Strategy**:
    - Design offer value and type based on {user_archetype} classification
    - Tailor messaging to match strategic objective for this archetype
    - Balance cost-effectiveness with conversion probability

    2. **Causal Persuasion Design**:
    - Address the specific barriers preventing conversion (for Persuadables)
    - Provide relationship reinforcement (for Sure Things)
    - Test engagement minimally (for Lost Causes)

    3. **Personalization**:
    - Reference user's specific spending patterns and preferences
    - Align with {domain} category and historical behavior
    - Use appropriate tone for their segment and lifecycle stage

    4. **Policy Compliance**:
    - Strictly adhere to ALL policies listed above
    - Ensure offer terms are clear and compliant

    **OUTPUT FORMAT (respond ONLY with valid JSON):**
    {{
    "offer_title": "Brief, compelling offer headline aligned with archetype strategy (max 60 chars)",
    "offer_description": "Full offer description that implements the archetype-specific strategy. For Persuadables, emphasize high value and urgency. For Sure Things, emphasize appreciation and relationship. For Lost Causes, keep simple and low-cost. (2-3 sentences)",
    "terms_and_conditions": "Clear, specific terms (validity period, spending requirements, exclusions)",
    "call_to_action": "Specific action user should take, framed appropriately for archetype",
    "offer_type": "points_multiplier|statement_credit|cashback|discount|exclusive_access",
    "offer_value": "Quantified benefit (e.g., '5X Points' for Persuadables, '2X Points' for Sure Things, '1.5X Points' for Lost Causes)",
    "expiration_days": 30-60 days (integer),
    "minimum_spend": Minimum spend requirement in dollars (integer, adjusted for archetype),
    "category_restrictions": ["list", "of", "specific", "eligible", "categories"],
    "featured_merchants": [
        {{
        "merchant_name": "Merchant Name",
        "merchant_category": "Specific category",
        "merchant_id": "MERCH_ID (from partner list above)",
        "merchant_type": "online|in-store|both"
        }}
    ],
    "reasoning": "2-3 sentence explanation of how this offer is specifically designed to persuade this {user_archetype}. Explain: (1) Why the offer value/type matches the archetype strategy, (2) How it addresses their specific behavioral barriers or situation, (3) Why this approach maximizes incremental impact or cost-effectiveness for this user type. Reference the expected uplift of {expected_uplift:+.1%}.",
    "policy_compliance_notes": ["list of policies verified and how offer complies"],
    "archetype_alignment": {{
        "target_archetype": "{user_archetype}",
        "strategy_applied": "Brief description of which strategy guideline was followed",
        "expected_incremental_value": "Estimated incremental revenue or engagement gain from this offer"
    }}
    }}

    **CRITICAL OFFER DESIGN RULES:**
    1. **Persuadables**: High-value offers (3X-5X points, $50-100 credits) with urgent, compelling messaging
    2. **Sure Things**: Modest offers (2X points, $25-50 credits) with appreciation-focused messaging
    3. **Lost Causes**: Minimal offers (1.5X-2X points, $10-25 credits) with simple re-engagement focus
    4. Select ONLY 2-4 merchants most relevant to user profile
    5. Ensure reasoning explicitly connects offer design to archetype classification
    6. Merchant selection should align with user's spending patterns in {domain}

    Generate the strategy-aligned offer now:"""
        
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
