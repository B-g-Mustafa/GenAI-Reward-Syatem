"""
Root Cause Analyzer - Hybrid approach combining rule-based metrics with LLM insights
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import json
from openai import OpenAI
from config import Config


class RootCauseAnalyzer:
    """Analyzes user spending behavior using hybrid rule-based + LLM approach"""
    
    def __init__(self, use_llm_insights: bool = True):
        """
        Initialize the analyzer
        
        Args:
            use_llm_insights: Whether to enhance analysis with LLM insights (default: True)
        """
        self.use_llm_insights = use_llm_insights
        
        if self.use_llm_insights:
            self.client = OpenAI(
                base_url=Config.OPENROUTER_BASE_URL,
                api_key=Config.OPENROUTER_API_KEY
            )
    
    def analyze_user_behavior(
        self, 
        user_data: Dict[str, Any],
        ml_recommendation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze user behavior and identify root causes for changes
        
        Args:
            user_data: User profile and transaction history
            ml_recommendation: ML model output (user_id, offer_flag, domain, confidence_score)
            
        Returns:
            Dict containing behavioral analysis and insights
        """
        analysis = {
            'user_id': user_data.get('user_id'),
            'domain': ml_recommendation.get('domain'),
            'confidence_score': ml_recommendation.get('confidence_score'),
            'insights': [],
            'spending_trends': {},
            'engagement_metrics': {},
            'risk_factors': []
        }
        
        # Analyze spending patterns
        spending_analysis = self._analyze_spending_patterns(
            user_data.get('transaction_history', []),
            ml_recommendation.get('domain')
        )
        analysis['spending_trends'] = spending_analysis
        
        # Analyze engagement behavior
        # Handle both flat structure (new) and nested structure (old)
        engagement_data = user_data.get('engagement_behavior', {})
        if not engagement_data:
            # Build from flat structure
            engagement_data = {
                'app_opens': user_data.get('total_app_opens_90d', 0),
                'offer_clicks': user_data.get('offer_clicks_90d', 0),
                'offers_shown': user_data.get('offer_views_90d', 0),
                'offers_accepted': user_data.get('offers_accepted_6m', 0),
                'redemptions': user_data.get('offers_accepted_6m', 0),  # Approximation
                'last_login': None  # Not available in flat structure
            }
        
        engagement_analysis = self._analyze_engagement(engagement_data)
        analysis['engagement_metrics'] = engagement_analysis
        
        # Analyze offer history
        # Handle both nested offer_history (old) and flat structure (new)
        offer_history = user_data.get('offer_history', [])
        if not offer_history:
            # Build summary from flat structure metrics
            offers_shown = user_data.get('offers_shown_6m', 0)
            offers_accepted = user_data.get('offers_accepted_6m', 0)
            acceptance_rate = user_data.get('historical_acceptance_rate', 0)
            last_offer_domain = user_data.get('last_offer_domain', '')
            preferred_domain_1 = user_data.get('preferred_domain_1', '')
            preferred_domain_2 = user_data.get('preferred_domain_2', '')
            
            # Create synthetic analysis from flat metrics
            offer_analysis = {
                'total_offers_received': offers_shown,
                'total_accepted': offers_accepted,
                'acceptance_rate': acceptance_rate * 100 if acceptance_rate <= 1 else acceptance_rate,
                'domain_performance': {},
                'preferred_domains': [d for d in [preferred_domain_1, preferred_domain_2] if d]
            }
            
            # Add last offer domain if available
            if last_offer_domain and last_offer_domain != 'N/A':
                offer_analysis['domain_performance'][last_offer_domain] = {
                    'total_offers': 1,
                    'accepted': 0,
                    'acceptance_rate': 0
                }
        else:
            offer_analysis = self._analyze_offer_history(offer_history)
        
        analysis['offer_performance'] = offer_analysis
        
        # Step 1: Generate rule-based insights (fast, quantitative)
        rule_based_insights = self._generate_rule_based_insights(
            spending_analysis, 
            engagement_analysis, 
            offer_analysis,
            user_data,
            ml_recommendation
        )
        analysis['insights'] = rule_based_insights
        analysis['rule_based_insights'] = rule_based_insights  # Keep for reference
        
        # Step 2: Enhance with LLM insights (contextual, qualitative)
        if self.use_llm_insights:
            try:
                llm_insights = self._generate_llm_insights(
                    user_data,
                    ml_recommendation,
                    spending_analysis,
                    engagement_analysis,
                    offer_analysis,
                    rule_based_insights
                )
                
                # Combine insights
                analysis['llm_insights'] = llm_insights.get('insights', [])
                analysis['llm_reasoning'] = llm_insights.get('reasoning', '')
                analysis['behavioral_patterns'] = llm_insights.get('patterns', [])
                
                # Merge for comprehensive view
                all_insights = rule_based_insights + llm_insights.get('insights', [])
                analysis['insights'] = all_insights
                
            except Exception as e:
                print(f"⚠️  LLM insights generation failed: {e}")
                print("   Falling back to rule-based insights only")
                # Keep rule-based insights as fallback
        
        return analysis
    
    def _analyze_spending_patterns(
        self, 
        transactions: List[Dict[str, Any]], 
        target_domain: str
    ) -> Dict[str, Any]:
        """Analyze spending patterns in the target domain"""
        
        if not transactions:
            return {
                'total_spend': 0,
                'transaction_count': 0,
                'average_transaction': 0,
                'trend': 'insufficient_data'
            }
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(transactions)
        
        # Ensure date column exists and is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        
        # Filter by domain/category if specified
        if target_domain and 'category' in df.columns:
            domain_df = df[df['category'].str.lower() == target_domain.lower()]
        else:
            domain_df = df
        
        if len(domain_df) == 0:
            return {
                'total_spend': 0,
                'transaction_count': 0,
                'average_transaction': 0,
                'trend': 'no_activity_in_domain',
                'recent_merchants': []
            }
        
        # Calculate metrics
        total_spend = domain_df['amount'].sum() if 'amount' in domain_df.columns else 0
        transaction_count = len(domain_df)
        avg_transaction = total_spend / transaction_count if transaction_count > 0 else 0
        
        # Analyze trend (last 60 days vs previous 60 days)
        trend_analysis = self._calculate_trend(domain_df)
        
        # Get frequent merchants
        recent_merchants = []
        if 'merchant' in domain_df.columns:
            merchant_counts = domain_df['merchant'].value_counts().head(5)
            recent_merchants = merchant_counts.to_dict()
        
        return {
            'total_spend': float(total_spend),
            'transaction_count': int(transaction_count),
            'average_transaction': float(avg_transaction),
            'trend': trend_analysis['trend'],
            'trend_percentage': trend_analysis['percentage_change'],
            'recent_merchants': recent_merchants,
            'last_transaction_date': domain_df['date'].max().isoformat() if 'date' in domain_df.columns else None
        }
    
    def _calculate_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate spending trend over time"""
        
        if 'date' not in df.columns or 'amount' not in df.columns:
            return {'trend': 'unknown', 'percentage_change': 0}
        
        now = datetime.now()
        last_60_days = now - timedelta(days=60)
        previous_60_days = now - timedelta(days=120)
        
        recent_spend = df[df['date'] >= last_60_days]['amount'].sum()
        previous_spend = df[(df['date'] >= previous_60_days) & (df['date'] < last_60_days)]['amount'].sum()
        
        if previous_spend == 0:
            if recent_spend > 0:
                return {'trend': 'increasing', 'percentage_change': 100}
            return {'trend': 'stable', 'percentage_change': 0}
        
        percentage_change = ((recent_spend - previous_spend) / previous_spend) * 100
        
        if percentage_change > 10:
            trend = 'increasing'
        elif percentage_change < -10:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'percentage_change': round(percentage_change, 2),
            'recent_spend': float(recent_spend),
            'previous_spend': float(previous_spend)
        }
    
    def _analyze_engagement(self, engagement_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user engagement with the app and offers"""
        
        return {
            'app_opens_per_month': engagement_data.get('app_opens', 0),
            'offer_click_rate': engagement_data.get('offer_clicks', 0) / max(engagement_data.get('offers_shown', 1), 1),
            'redemption_rate': engagement_data.get('redemptions', 0) / max(engagement_data.get('offers_accepted', 1), 1),
            'last_login': engagement_data.get('last_login'),
            'engagement_level': self._classify_engagement_level(engagement_data)
        }
    
    def _classify_engagement_level(self, engagement_data: Dict[str, Any]) -> str:
        """Classify user engagement level"""
        
        app_opens = engagement_data.get('app_opens', 0)
        offer_clicks = engagement_data.get('offer_clicks', 0)
        
        if app_opens >= 20 and offer_clicks >= 5:
            return 'high'
        elif app_opens >= 10 and offer_clicks >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _analyze_offer_history(self, offer_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze past offer performance"""
        
        if not offer_history:
            return {
                'total_offers_received': 0,
                'acceptance_rate': 0,
                'preferred_domains': []
            }
        
        total_offers = len(offer_history)
        accepted_offers = sum(1 for offer in offer_history if offer.get('status') == 'accepted')
        
        # Count by domain
        domain_counts = {}
        domain_acceptances = {}
        
        for offer in offer_history:
            domain = offer.get('domain', 'unknown')
            status = offer.get('status')
            
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            if status == 'accepted':
                domain_acceptances[domain] = domain_acceptances.get(domain, 0) + 1
        
        # Calculate acceptance rate by domain
        domain_performance = {}
        for domain, count in domain_counts.items():
            acceptance_rate = (domain_acceptances.get(domain, 0) / count) * 100
            domain_performance[domain] = {
                'total_offers': count,
                'accepted': domain_acceptances.get(domain, 0),
                'acceptance_rate': round(acceptance_rate, 2)
            }
        
        # Sort domains by acceptance rate
        preferred_domains = sorted(
            domain_performance.items(), 
            key=lambda x: x[1]['acceptance_rate'], 
            reverse=True
        )
        
        return {
            'total_offers_received': total_offers,
            'total_accepted': accepted_offers,
            'acceptance_rate': round((accepted_offers / total_offers * 100), 2) if total_offers > 0 else 0,
            'domain_performance': domain_performance,
            'preferred_domains': [d[0] for d in preferred_domains[:3]]
        }
    
    def _generate_rule_based_insights(
        self,
        spending_analysis: Dict[str, Any],
        engagement_analysis: Dict[str, Any],
        offer_analysis: Dict[str, Any],
        user_data: Dict[str, Any],
        ml_recommendation: Dict[str, Any]
    ) -> List[str]:
        """Generate rule-based human-readable insights from quantitative analysis"""
        
        insights = []
        domain = ml_recommendation.get('domain', 'general')
        
        # Extract user profile data (now flat structure)
        preferred_domain_1 = user_data.get('preferred_domain_1', '')
        preferred_domain_2 = user_data.get('preferred_domain_2', '')
        
        # Spending insights
        trend = spending_analysis.get('trend')
        percentage_change = spending_analysis.get('trend_percentage', 0)
        
        if trend == 'decreasing':
            insights.append(
                f"User's {domain} spending has decreased by {abs(percentage_change):.1f}% "
                f"in the last 60 days compared to the previous period."
            )
        elif trend == 'increasing':
            insights.append(
                f"User shows increasing interest in {domain} with spending up "
                f"{percentage_change:.1f}% in recent months."
            )
        elif trend == 'no_activity_in_domain':
            insights.append(
                f"User has no recent activity in {domain}, presenting an opportunity "
                f"to introduce them to this category."
            )
        
        # Engagement insights
        engagement_level = engagement_analysis.get('engagement_level')
        if engagement_level == 'high':
            insights.append(
                "User is highly engaged with the app and frequently interacts with offers."
            )
        elif engagement_level == 'low':
            insights.append(
                "User has low engagement - a compelling offer could re-activate their interest."
            )
        
        # Offer history insights
        acceptance_rate = offer_analysis.get('acceptance_rate', 0)
        if acceptance_rate > 50:
            insights.append(
                f"User has a strong history of accepting offers ({acceptance_rate:.0f}% acceptance rate)."
            )
        elif acceptance_rate < 20 and offer_analysis.get('total_offers_received', 0) > 3:
            insights.append(
                f"User has been selective with past offers ({acceptance_rate:.0f}% acceptance rate) - "
                f"ensure this offer is highly relevant."
            )
        
        # Domain preference insights from user profile
        preferred_domain_1 = user_data.get('preferred_domain_1', '')
        preferred_domain_2 = user_data.get('preferred_domain_2', '')
        preferred_domains_user = [d for d in [preferred_domain_1, preferred_domain_2] if d]
        
        # Check both profile and historical offer preferences
        preferred_domains_offer = offer_analysis.get('preferred_domains', [])
        
        if domain.lower() in [d.lower() for d in preferred_domains_user]:
            insights.append(
                f"{domain} is one of user's preferred categories based on their spending profile."
            )
        elif domain in preferred_domains_offer:
            insights.append(
                f"{domain} is among user's preferred categories based on past offer acceptance."
            )
        
        # Merchant insights
        recent_merchants = spending_analysis.get('recent_merchants', {})
        if recent_merchants:
            top_merchant = list(recent_merchants.keys())[0] if recent_merchants else None
            if top_merchant:
                insights.append(
                    f"User frequently shops at {top_merchant} - consider merchant-specific offers."
                )
        
        return insights
    
    def _generate_llm_insights(
        self,
        user_data: Dict[str, Any],
        ml_recommendation: Dict[str, Any],
        spending_analysis: Dict[str, Any],
        engagement_analysis: Dict[str, Any],
        offer_analysis: Dict[str, Any],
        rule_based_insights: List[str]
    ) -> Dict[str, Any]:
        """
        Generate enhanced insights using LLM analysis
        
        Returns:
            Dict with 'insights', 'reasoning', and 'patterns'
        """
        
        # Build a comprehensive context for the LLM
        prompt = self._build_llm_analysis_prompt(
            user_data,
            ml_recommendation,
            spending_analysis,
            engagement_analysis,
            offer_analysis,
            rule_based_insights
        )
        
        try:
            response = self.client.chat.completions.create(
                model=Config.OPENROUTER_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert financial behavior analyst for American Express. Analyze user spending patterns and provide deep, actionable insights about their behavior. Be specific, insightful, and focus on understanding the 'why' behind behavioral changes."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Parse the LLM response
            llm_output = response.choices[0].message.content
            
            # Try to extract JSON if structured, otherwise parse as text
            try:
                # Look for JSON in the response
                start_idx = llm_output.find('{')
                end_idx = llm_output.rfind('}') + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    json_str = llm_output[start_idx:end_idx]
                    parsed_output = json.loads(json_str)
                else:
                    # Fallback: parse as plain text
                    parsed_output = self._parse_llm_text_output(llm_output)
                
                return {
                    'insights': parsed_output.get('insights', []),
                    'reasoning': parsed_output.get('reasoning', ''),
                    'patterns': parsed_output.get('patterns', []),
                    'raw_output': llm_output
                }
                
            except json.JSONDecodeError:
                # Fallback: parse as plain text
                parsed_output = self._parse_llm_text_output(llm_output)
                return {
                    'insights': parsed_output.get('insights', []),
                    'reasoning': parsed_output.get('reasoning', llm_output),
                    'patterns': parsed_output.get('patterns', []),
                    'raw_output': llm_output
                }
                
        except Exception as e:
            print(f"LLM analysis error: {e}")
            return {
                'insights': [],
                'reasoning': '',
                'patterns': [],
                'error': str(e)
            }
    
    def _build_llm_analysis_prompt(
        self,
        user_data: Dict[str, Any],
        ml_recommendation: Dict[str, Any],
        spending_analysis: Dict[str, Any],
        engagement_analysis: Dict[str, Any],
        offer_analysis: Dict[str, Any],
        rule_based_insights: List[str]
    ) -> str:
        """Build a comprehensive prompt for LLM analysis"""
        
        # Extract flat user data structure
        name = user_data.get('name', 'Unknown')
        segment = user_data.get('segment', 'Standard')
        tenure_years = user_data.get('tenure_years', 0)
        age = user_data.get('age', 'N/A')
        gender = user_data.get('gender', 'N/A')
        location = user_data.get('location', 'N/A')
        persona = user_data.get('persona', 'N/A')
        customer_lifecycle_stage = user_data.get('customer_lifecycle_stage', 'N/A')
        churn_risk_score = user_data.get('churn_risk_score', 0)
        
        # Get transaction history if available
        transactions = user_data.get('transaction_history', [])
        domain = ml_recommendation.get('domain', 'general')
        
        # Format recent transactions
        recent_txns = sorted(
            transactions,
            key=lambda x: x.get('date', ''),
            reverse=True
        )[:10]
        
        txn_summary = "\n".join([
            f"  - {t.get('date', 'N/A')}: ${t.get('amount', 0):.2f} at {t.get('merchant', 'Unknown')} ({t.get('category', 'N/A')})"
            for t in recent_txns
        ]) if recent_txns else "No recent transactions available"
        
        prompt = f"""Analyze this American Express card member's behavior and provide deep insights:

**USER PROFILE:**
- Name: {name}
- Segment: {segment}
- Account Age: {tenure_years} years
- Age: {age}
- Gender: {gender}
- Location: {location}
- Persona: {persona}
- Lifecycle Stage: {customer_lifecycle_stage}
- Churn Risk Score: {churn_risk_score}

**ML RECOMMENDATION:**
- Target Category: {domain}
- Confidence: {ml_recommendation.get('confidence_score', 0):.1%}

**QUANTITATIVE METRICS (Rule-Based Analysis):**

Spending in {domain}:
- Total Spend: ${spending_analysis.get('total_spend', 0):,.2f}
- Transaction Count: {spending_analysis.get('transaction_count', 0)}
- Trend: {spending_analysis.get('trend', 'unknown')} ({spending_analysis.get('trend_percentage', 0):+.1f}%)
- Average Transaction: ${spending_analysis.get('average_transaction', 0):.2f}

Engagement:
- Level: {engagement_analysis.get('engagement_level', 'unknown')}
- Offer Click Rate: {engagement_analysis.get('offer_click_rate', 0):.1%}
- Redemption Rate: {engagement_analysis.get('redemption_rate', 0):.1%}

Offer History:
- Total Offers: {offer_analysis.get('total_offers_received', 0)}
- Acceptance Rate: {offer_analysis.get('acceptance_rate', 0):.1f}%
- Preferred Domains: {', '.join(offer_analysis.get('preferred_domains', [])[:3])}

**RECENT TRANSACTIONS:**
{txn_summary}

**RULE-BASED INSIGHTS (Already Generated):**
{chr(10).join(f"- {insight}" for insight in rule_based_insights)}

**YOUR TASK:**
Go beyond the quantitative metrics. Provide:

1. **Deep Behavioral Insights** (3-5 insights):
   - Why might spending patterns have changed?
   - What life events or circumstances could explain this?
   - What motivations or preferences are evident?
   - Any seasonal or temporal patterns?

2. **Psychological/Behavioral Patterns** (2-3 patterns):
   - Shopping behavior patterns
   - Decision-making style
   - Value drivers (convenience, luxury, savings, etc.)

3. **Root Cause Reasoning**:
   - What's the real "why" behind the ML recommendation?
   - Connect the dots between different data points
   - Identify non-obvious correlations

**OUTPUT FORMAT (JSON):**
{{
    "insights": [
        "Insight 1 with specific reasoning...",
        "Insight 2 with specific reasoning...",
        "Insight 3 with specific reasoning..."
    ],
    "patterns": [
        "Pattern 1: description...",
        "Pattern 2: description..."
    ],
    "reasoning": "Comprehensive 2-3 sentence explanation connecting all observations and explaining why this user is a good target for {domain} offers"
}}

Generate the analysis now:"""
        
        return prompt
    
    def _parse_llm_text_output(self, text: str) -> Dict[str, Any]:
        """Parse LLM output when it's not in JSON format"""
        
        insights = []
        patterns = []
        reasoning = ""
        
        # Simple parsing - look for bullet points or numbered lists
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith(('- ', '• ', '* ', '1.', '2.', '3.')):
                # Remove the bullet/number
                clean_line = line.lstrip('- •*123456789. ')
                if clean_line:
                    insights.append(clean_line)
        
        # If no structured insights found, use the whole text as reasoning
        if not insights:
            reasoning = text
        else:
            # Use first few sentences as reasoning
            sentences = text.split('.')
            reasoning = '. '.join(sentences[:3]) + '.' if sentences else text
        
        return {
            'insights': insights[:5],  # Limit to 5
            'patterns': patterns,
            'reasoning': reasoning
        }

