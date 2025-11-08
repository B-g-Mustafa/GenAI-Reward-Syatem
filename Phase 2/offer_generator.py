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

    **FEW-SHOT EXAMPLES:**

    Below are two canonical examples demonstrating strategic offer design:

    ---

    **EXAMPLE 1: OFFER FOR PERSUADABLE USER**

    Input Context:
    - Archetype: Persuadable (Expected Uplift: +43%)
    - Category: Dining, Current spend: $2,450 (declining -18%)
    - Segment: Gold, Historical acceptance: 60%

    Generated Offer:

    {{
    "offer_title": "Earn 5X Points on Dining - Limited Time!",
    "offer_description": "We've noticed you enjoy great dining experiences. For the next 45 days, earn 5X Membership Rewards points at restaurants, cafes, and bars. That's up to $150 in point value on your typical quarterly dining spend. Perfect timing to explore new restaurants or revisit your favorites while maximizing rewards.",
    "terms_and_conditions": "Offer valid for 45 days from activation. Earn 5X points (instead of standard 1X) on eligible dining purchases. Minimum spend: $400 within offer period. Maximum bonus points: 15,000. Excludes grocery stores. Points post within 2 billing cycles.",
    "call_to_action": "Activate your 5X Dining Points offer now and start earning bonus rewards on your next meal out!",
    "offer_type": "points_multiplier",
    "offer_value": "5X Points (4X bonus)",
    "expiration_days": 45,
    "minimum_spend": 400,
    "category_restrictions": ["dining", "restaurants", "cafes", "bars", "food_delivery"],
    "featured_merchants": [
    {{
    "merchant_name": "OpenTable Restaurants",
    "merchant_category": "Fine Dining",
    "merchant_id": "MERCH_DINING_001",
    "merchant_type": "both"
    }},
    {{
    "merchant_name": "DoorDash",
    "merchant_category": "Food Delivery",
    "merchant_id": "MERCH_DELIVERY_003",
    "merchant_type": "online"
    }}
    ],
    "reasoning": "This Persuadable user requires a compelling high-value offer (5X points vs standard 1X) to overcome their 18% spending decline and reverse disengagement. The 45-day window creates urgency, while the $400 minimum spend threshold is achievable yet meaningful. The 43% expected uplift justifies this premium offer - incremental revenue far exceeds points cost. Strategic focus on reversing negative trend through tangible rewards that reactivate dormant dining patterns.",
    "policy_compliance_notes": [
    "Points multiplier complies with rewards policy maximum of 5X",
    "45-day expiration within policy range of 30-60 days"
    ],
    "archetype_alignment": {{
    "target_archetype": "Persuadable",
    "strategy_applied": "High-value offer with urgency to overcome inertia and reverse declining engagement",
    "expected_incremental_value": "Estimated $1,050 incremental dining spend (43% uplift), generating $42 incremental revenue"
    }}
    }}

    ---

    **EXAMPLE 2: OFFER FOR SURE THING USER**

    Input Context:
    - Archetype: Sure Thing (Expected Uplift: +4%)
    - Category: Grocery, Current spend: $4,200 (growing +15%)
    - Segment: Platinum, Historical acceptance: 95%

    Generated Offer:

    {{
    "offer_title": "Platinum Member Appreciation: 2X Grocery Points",
    "offer_description": "Thank you for being a valued Platinum Card Member. As a token of our appreciation for your continued loyalty, enjoy 2X points on your grocery purchases for the next 60 days. Continue shopping at your favorite stores while earning enhanced rewards on everyday essentials.",
    "terms_and_conditions": "Offer valid for 60 days from activation. Earn 2X points (instead of standard 1X) on eligible grocery store purchases. No minimum spend required. Maximum bonus points: 5,000. Points post within 2 billing cycles.",
    "call_to_action": "Activate your Platinum Appreciation offer to start earning 2X points on groceries.",
    "offer_type": "points_multiplier",
    "offer_value": "2X Points (1X bonus)",
    "expiration_days": 60,
    "minimum_spend": 0,
    "category_restrictions": ["grocery_stores", "supermarkets", "grocery_delivery"],
    "featured_merchants": [
    {{
    "merchant_name": "Whole Foods Market",
    "merchant_category": "Premium Grocery",
    "merchant_id": "MERCH_GROCERY_001",
    "merchant_type": "in-store"
    }},
    {{
    "merchant_name": "Instacart",
    "merchant_category": "Grocery Delivery",
    "merchant_id": "MERCH_GROCERY_DEL_002",
    "merchant_type": "online"
    }}
    ],
    "reasoning": "This Sure Thing user has only 4% incremental uplift potential, so a premium offer would waste resources on guaranteed behavior. The modest 2X points multiplier (vs 5X for Persuadables) provides relationship appreciation without excessive cost. No minimum spend requirement respects Platinum status. The 60-day window offers flexibility without urgency, as urgency is irrelevant for users with 88% baseline conversion. Cost-effective approach maintains satisfaction on inevitable transactions while preserving profit margins.",
    "policy_compliance_notes": [
    "2X points multiplier within policy limits",
    "No minimum spend aligns with Platinum benefits policy"
    ],
    "archetype_alignment": {{
    "target_archetype": "Sure Thing",
    "strategy_applied": "Low-cost loyalty appreciation that maintains satisfaction without overspending on guaranteed conversions",
    "expected_incremental_value": "Minimal incremental spend ($170, 4% uplift) but strengthens relationship at low cost"
    }}
    }}

    ---

    **KEY DIFFERENCES:**
    - Persuadable: 5X points, 45 days, $400 minimum, urgent messaging
    - Sure Thing: 2X points, 60 days, no minimum, appreciation messaging

    ---

    **OUTPUT FORMAT (respond ONLY with valid JSON):**
    {{
    "offer_title": "Brief, compelling offer headline aligned with archetype strategy (max 60 chars)",
    "offer_description": "Full offer description implementing archetype-specific strategy (2-3 sentences)",
    "terms_and_conditions": "Clear, specific terms",
    "call_to_action": "Specific action framed for archetype",
    "offer_type": "points_multiplier|statement_credit|cashback|discount|exclusive_access",
    "offer_value": "Quantified benefit",
    "expiration_days": 30-60 (integer),
    "minimum_spend": dollars (integer),
    "category_restrictions": ["list", "of", "categories"],
    "featured_merchants": [
        {{
        "merchant_name": "Name",
        "merchant_category": "Category",
        "merchant_id": "MERCH_ID",
        "merchant_type": "online|in-store|both"
        }}
    ],
    "reasoning": "2-3 sentences explaining how this offer persuades this {user_archetype}. Include: (1) Why value/type matches archetype strategy, (2) How it addresses behavioral barriers, (3) Why this maximizes incremental impact for expected uplift of {expected_uplift:+.1%}.",
    "policy_compliance_notes": ["list of verified policies"],
    "archetype_alignment": {{
        "target_archetype": "{user_archetype}",
        "strategy_applied": "Brief strategy description",
        "expected_incremental_value": "Estimated gain from offer"
    }}
    }}

    **CRITICAL RULES:**
    1. Persuadables: High-value (3X-5X points, $50-100 credits), urgent messaging
    2. Sure Things: Modest (2X points, $25-50 credits), appreciation messaging
    3. Lost Causes: Minimal (1.5X-2X points, $10-25 credits), simple re-engagement
    4. Select ONLY 2-4 merchants most relevant to user

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
