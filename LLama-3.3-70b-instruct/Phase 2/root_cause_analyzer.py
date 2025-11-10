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
        """Build a comprehensive prompt for causal analysis with archetype classification"""
        
        # Extract flat user data structure
        name = user_data.get('name', 'Unknown')
        segment = user_data.get('segment', 'Standard')
        tenure_years = user_data.get('tenure_years', 0)
        age = user_data.get('age', 'N/A')
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
        
        prompt = f"""You are an expert causal analyst for American Express specializing in uplift modeling and treatment effect estimation. Your task is to determine the probable incremental impact of sending an offer to this card member.

    **CAUSAL INFERENCE FRAMEWORK:**
    Your analysis must classify users into one of three causal archetypes based on uplift modeling principles:

    1. **PERSUADABLES** - Users whose behavior will change positively due to the offer
    - Would NOT convert without the offer, but WILL convert with it
    - High incremental treatment effect (positive uplift)
    - These are the target segment for offers

    2. **SURE THINGS** - Users who will convert regardless of intervention
    - Will convert even WITHOUT the offer
    - Near-zero or slightly negative treatment effect
    - Offering may waste resources without incremental benefit

    3. **LOST CAUSES** - Users unlikely to respond despite intervention
    - Will NOT convert even WITH the offer
    - Near-zero treatment effect
    - Intervention has minimal to no impact

    **USER PROFILE:**
    - Name: {name}
    - Segment: {segment}
    - Account Age: {tenure_years} years
    - Age: {age}
    - Lifecycle Stage: {customer_lifecycle_stage}
    - Churn Risk Score: {churn_risk_score}

    **ML RECOMMENDATION:**
    - Target Category: {domain}
    - Confidence: {ml_recommendation.get('confidence_score', 0):.1%}

    **QUANTITATIVE METRICS:**

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

    **RULE-BASED INSIGHTS:**
    {chr(10).join(f"- {insight}" for insight in rule_based_insights)}

    **YOUR CAUSAL ANALYSIS TASK:**

    1. **Counterfactual Reasoning** (Critical):
    - Hypothesize: What would this user's behavior be if NO offer is sent?
    - Consider: What is the baseline probability of conversion without intervention?
    - Estimate: How much would an offer incrementally change their likelihood to convert?

    2. **Classify into Causal Archetype**:
    Based on evidence, determine which archetype best describes this user:
    
    **Indicators of PERSUADABLES:**
    - Declining engagement but past positive responses
    - Moderate spending with recent downward trend
    - Category interest shown but needs activation
    - Historical offer acceptance but selective behavior
    - Medium churn risk with recoverable engagement
    
    **Indicators of SURE THINGS:**
    - Consistently high engagement regardless of offers
    - Strong upward spending trend in target category
    - High organic transaction frequency
    - Low churn risk with stable patterns
    - Will likely transact even without incentive
    
    **Indicators of LOST CAUSES:**
    - Very low engagement despite past offers
    - Near-zero spending in target category
    - High churn risk with dormant account
    - Consistent non-response to multiple offers
    - No behavioral signals suggesting receptiveness

    3. **Treatment Effect Estimation**:
    - What is the expected lift (percentage point increase) from sending an offer?
    - Would the user convert without the offer? (baseline probability)
    - Would the user convert with the offer? (treatment probability)
    - Incremental effect = Treatment probability - Baseline probability

    4. **Provide Justification**:
    - Connect specific data points to your archetype classification
    - Explain the causal mechanism: WHY would/wouldn't an offer change behavior?
    - Consider confounding factors and alternative explanations
    - Assess confidence in your classification

    **FEW-SHOT EXAMPLES:**

    Below are three high-quality examples demonstrating the expected causal analysis format:

    ---

    **EXAMPLE 1: PERSUADABLE USER**

    User Profile Summary:
    - Segment: Gold, Tenure: 3 years, Churn Risk: 0.45
    - Dining spending: $2,450 (down 18% from last quarter)
    - Engagement: Medium, Offer acceptance: 60%

    Analysis Output:

    {{
    "causal_analysis": {{
    "user_archetype": "Persuadable",
    "confidence_score": 0.82,
    "baseline_conversion_probability": 0.25,
    "treatment_conversion_probability": 0.68,
    "expected_uplift": 0.43,
    "justification": "This user exhibits classic Persuadable characteristics: declining dining engagement despite historical interest, moderate churn risk, and selective but positive offer response history. Without intervention, the 18% spending decline suggests continued disengagement (baseline 25% conversion). However, their 60% historical acceptance rate indicates receptiveness to well-targeted incentives. A compelling dining offer can reverse the negative trend by reactivating dormant behavior patterns.",
    "key_signals": [
    "Declining spending (-18%) in previously active category indicates recoverable disengagement",
    "60% historical offer acceptance shows strong treatment responsiveness",
    "Moderate churn risk (0.45) suggests user is at decision point, not fully disengaged"
    ],
    "counterfactual_scenario": "Without an offer, this user will likely continue their dining spending decline, potentially shifting budget to competing cards. The downward trajectory suggests baseline conversion probability of only 25% for increased dining spend in the next 30 days."
    }},
    "insights": [
    "Recent travel offer acceptance suggests user responds to category-specific, high-value propositions",
    "Declining dining spend coincides with competitor enhanced rewards - user may be shifting wallet share",
    "Historical 60% acceptance rate indicates strong treatment effect potential when offers match priorities"
    ],
    "patterns": [
    "Category rotation behavior: shifts focus between dining and travel quarterly",
    "Offer selectivity increased over tenure: earlier 80% vs current 60% acceptance"
    ],
    "reasoning": "This Persuadable user requires targeted intervention. The 43% expected uplift stems from high treatment responsiveness combined with current disengagement. A well-designed dining offer can recapture wallet share before spending habits fully shift to competitors."
    }}

    ---

    **EXAMPLE 2: SURE THING USER**

    User Profile Summary:
    - Segment: Platinum, Tenure: 7 years, Churn Risk: 0.12
    - Grocery spending: $4,200 (up 15% consistently over 6 months)
    - Engagement: Very High, Offer acceptance: 95%

    Analysis Output:

    {{
    "causal_analysis": {{
    "user_archetype": "Sure Thing",
    "confidence_score": 0.91,
    "baseline_conversion_probability": 0.88,
    "treatment_conversion_probability": 0.92,
    "expected_uplift": 0.04,
    "justification": "This user demonstrates Sure Thing behavior with extremely high baseline conversion probability (88%). Consistent 15% spending growth over six months occurs organically, independent of offer receipt. The 95% offer acceptance rate reflects general engagement rather than treatment-dependent behavior. The minimal 4% incremental uplift indicates offers provide marginal value beyond relationship maintenance.",
    "key_signals": [
    "Sustained 15% spending growth over 6 months with or without active offers",
    "Very low churn risk (0.12) and 7-year tenure indicate deep loyalty",
    "95% offer acceptance but spending patterns unchanged by offer timing"
    ],
    "counterfactual_scenario": "Without any offer, this user will continue their current grocery spending trajectory, maintaining the 15% growth rate. Historical data shows no correlation between offer timing and spending spikes - behavior is driven by lifestyle needs, not promotional incentives."
    }},
    "insights": [
    "Grocery spending growth correlates with life stage changes (recent home purchase) rather than promotions",
    "User accepts nearly all offers but spending patterns show no causal link to offer receipt timing",
    "Platinum segment loyalty indicates intrinsic brand commitment independent of incentives"
    ],
    "patterns": [
    "Consistent month-over-month spending increase regardless of promotional calendar",
    "Near-perfect offer acceptance (95%) but zero deviation in spending following redemption"
    ],
    "reasoning": "This Sure Thing user requires minimal investment. With only 4% incremental uplift potential, expensive offers would waste resources on guaranteed behavior. Low-cost appreciation gestures maintain satisfaction while preserving profit margins."
    }}

    ---

    **EXAMPLE 3: LOST CAUSE USER**

    User Profile Summary:
    - Segment: Standard, Tenure: 5 years, Churn Risk: 0.82
    - Entertainment spending: $180 (down 65% over 12 months)
    - Engagement: Very Low, Offer acceptance: 8%

    Analysis Output:

    {{
    "causal_analysis": {{
    "user_archetype": "Lost Cause",
    "confidence_score": 0.87,
    "baseline_conversion_probability": 0.05,
    "treatment_conversion_probability": 0.08,
    "expected_uplift": 0.03,
    "justification": "This user exhibits Lost Cause characteristics with minimal response probability despite intervention. The 65% spending decline over 12 months, combined with 0% response to the last 12 consecutive offers and 90-day dormancy, indicates deep disengagement beyond simple promotional recovery. High churn risk (0.82) and only 8% historical acceptance rate suggest structural barriers that offers cannot overcome.",
    "key_signals": [
    "Zero transactions in 90 days and ignored 12 consecutive offers shows complete disengagement",
    "65% spending decline over 12 months indicates permanent behavioral shift",
    "8% lifetime offer acceptance with declining trend suggests fundamental lack of treatment responsiveness"
    ],
    "counterfactual_scenario": "Without an offer, this user will remain dormant with 95% probability. Their entertainment spending has shifted permanently to alternative payment methods, likely driven by competitor switching or life changes that promotional offers cannot reverse."
    }},
    "insights": [
    "Extended dormancy indicates user has established alternative payment patterns beyond recovery",
    "Entertainment spending decline coincides with broader account inactivity",
    "High churn risk reflects behavioral reality: user has mentally churned even if account remains open"
    ],
    "patterns": [
    "Progressive disengagement: acceptance declined from 15% (years 1-3) to 0% (recent)",
    "No response to varied offer types suggests offers are irrelevant to current needs"
    ],
    "reasoning": "This Lost Cause user shows minimal uplift potential (3%) with structural disengagement beyond promotional recovery. Significant investment would yield negligible returns. A minimal diagnostic offer can test for residual responsiveness, but expectations should remain low."
    }}

    ---

    **OUTPUT FORMAT (respond ONLY with valid JSON):**
    {{
    "causal_analysis": {{
        "user_archetype": "Persuadable|Sure Thing|Lost Cause",
        "confidence_score": 0.0-1.0,
        "baseline_conversion_probability": 0.0-1.0,
        "treatment_conversion_probability": 0.0-1.0,
        "expected_uplift": -1.0 to 1.0,
        "justification": "Comprehensive 3-4 sentence explanation connecting observed behavior to archetype classification",
        "key_signals": [
        "Signal 1 supporting classification",
        "Signal 2 supporting classification",
        "Signal 3 supporting classification"
        ],
        "counterfactual_scenario": "2-3 sentence description of what would likely happen if NO offer is sent"
    }},
    "insights": [
        "Causal insight 1 with treatment effect reasoning",
        "Causal insight 2 with counterfactual thinking",
        "Causal insight 3 with uplift estimation"
    ],
    "patterns": [
        "Behavioral pattern 1 relevant to treatment response",
        "Behavioral pattern 2 relevant to treatment response"
    ],
    "reasoning": "2-3 sentence summary explaining the probable incremental impact of an offer on this specific user"
    }}

    Generate the causal analysis now:"""
        
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

