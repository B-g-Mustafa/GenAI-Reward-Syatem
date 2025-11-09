"""
Demo Script - Demonstrates the GenAI Reasoner and Offer Generator

Run this script to test the complete system with sample data
"""

import os
from genai_reasoner import GenAIReasoner
from sample_data import get_sample_data
import json


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def print_offer_summary(response: dict):
    """Print a formatted summary of the offer"""
    
    if response.get('recommendation') == 'no_offer':
        print("❌ No offer recommended")
        print(f"Reason: {response.get('reason')}")
        return
    
    if not response.get('offer', {}).get('success'):
        print("❌ Offer generation failed")
        print(f"Error: {response.get('offer', {}).get('error')}")
        return
    
    offer = response['offer']
    analysis = response['behavioral_analysis']
    
    print("✅ OFFER SUCCESSFULLY GENERATED\n")
    
    print("📊 RULE-BASED INSIGHTS:")
    for insight in analysis.get('rule_based_insights', [])[:3]:
        print(f"  • {insight}")
    
    # Show LLM insights if available
    if analysis.get('llm_insights'):
        print(f"\n🧠 LLM-ENHANCED INSIGHTS:")
        for insight in analysis.get('llm_insights', [])[:3]:
            print(f"  • {insight}")
        
        if analysis.get('llm_reasoning'):
            print(f"\n💡 LLM REASONING:")
            print(f"  {analysis.get('llm_reasoning')}")
        
        if analysis.get('behavioral_patterns'):
            print(f"\n🔍 BEHAVIORAL PATTERNS:")
            for pattern in analysis.get('behavioral_patterns', [])[:2]:
                print(f"  • {pattern}")
    
    print(f"\n🎯 OFFER DETAILS:")
    print(f"  Title: {offer.get('offer_title')}")
    print(f"  Value: {offer.get('offer_value')}")
    print(f"  Type: {offer.get('offer_type')}")
    
    print(f"\n📝 DESCRIPTION:")
    print(f"  {offer.get('offer_description')}")
    
    print(f"\n✅ CALL TO ACTION:")
    print(f"  {offer.get('call_to_action')}")
    
    print(f"\n📋 TERMS:")
    print(f"  • {offer.get('terms_and_conditions')}")
    if offer.get('minimum_spend'):
        print(f"  • Minimum spend: ${offer.get('minimum_spend')}")
    if offer.get('expiration_days'):
        print(f"  • Valid for: {offer.get('expiration_days')} days")
    
    print(f"\n🤔 OFFER REASONING:")
    print(f"  {offer.get('reasoning')}")
    
    print(f"\n✓ Policy Compliance:")
    for policy in offer.get('policy_compliance_notes', [])[:3]:
        print(f"  • {policy}")


def demo_single_user(reasoner: GenAIReasoner, user_id: str = 'USR001'):
    """Demonstrate offer generation for a single user"""
    
    print_section(f"DEMO: Single User Offer Generation - {user_id}")
    
    # Get sample data
    user_data, ml_recommendation = get_sample_data(user_id)
    
    if not user_data or not ml_recommendation:
        print(f"❌ Sample data not found for {user_id}")
        return None
    
    # Display input data - handle both flat and nested structures
    print("📥 INPUT DATA:")
    user_name = user_data.get('name') or user_data.get('profile', {}).get('name', 'Unknown')
    user_segment = user_data.get('segment') or user_data.get('profile', {}).get('segment', 'Standard')
    print(f"  User: {user_name} ({user_id})")
    print(f"  Segment: {user_segment}")
    print(f"  ML Recommendation: {ml_recommendation['domain']} "
          f"(confidence: {ml_recommendation['confidence_score']:.1%})")
    print(f"  Transaction History: {len(user_data['transaction_history'])} transactions")
    print(f"  Offer History: {len(user_data['offer_history'])} previous offers")
    
    # Process the recommendation
    print("\n🔄 PROCESSING...")
    response = reasoner.process_ml_recommendation(user_data, ml_recommendation)
    
    # Display results
    print_section("RESULTS")
    print_offer_summary(response)
    
    return response


def demo_all_users(reasoner: GenAIReasoner):
    """Demonstrate offer generation for all sample users"""
    
    print_section("DEMO: Batch Processing - All Users")
    
    samples = get_sample_data()
    results = []
    
    for i, (user_data, ml_recommendation) in enumerate(samples, 1):
        print(f"\n{'─'*80}")
        print(f"Processing User {i}/3: {user_data['user_id']} - {user_data['profile']['name']}")
        print(f"{'─'*80}")
        
        response = reasoner.process_ml_recommendation(user_data, ml_recommendation)
        results.append(response)
        
        # Brief summary
        if response.get('offer', {}).get('success'):
            print(f"✅ {response['offer']['offer_title']}")
            print(f"   {response['offer']['offer_value']}")
        else:
            print("❌ Offer generation failed")
    
    return results


def demo_policy_system(reasoner: GenAIReasoner):
    """Demonstrate the policy RAG system"""
    
    print_section("DEMO: Policy Retrieval System")
    
    # Get policy summary
    policies = reasoner.get_policy_summary()
    
    print(f"📚 Loaded Policies: {len(policies)}\n")
    
    for policy in policies:
        print(f"  • {policy['id']}")
        print(f"    Type: {policy['type']} | Domain: {policy['domain']}")
        print(f"    Preview: {policy['preview']}\n")


def demo_custom_policy(reasoner: GenAIReasoner):
    """Demonstrate adding a custom policy"""
    
    print_section("DEMO: Adding Custom Policy")
    
    custom_policy = """
    Holiday Bonus Policy:
    - Special holiday promotions: November-December
    - Maximum bonus: 5X points on select categories
    - Limited time offers: 14-30 days validity
    - Bonus categories: Gift purchases, dining, travel
    - Spending cap: $10,000 per promotion period
    """
    
    reasoner.add_custom_policy(
        policy_id='holiday_bonus_2025',
        content=custom_policy,
        policy_type='seasonal',
        domain='all'
    )
    
    print("✅ Custom policy added successfully")


def export_results(response: dict, user_id: str, output_dir: str = './output'):
    """Export results to files"""
    
    print_section("EXPORTING RESULTS")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Export JSON
    json_path = os.path.join(output_dir, f'offer_{user_id}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(response, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON exported: {json_path}")
    
    # Export Email HTML
    if response.get('email_html'):
        html_path = os.path.join(output_dir, f'offer_{user_id}_email.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(response['email_html'])
        print(f"✅ Email HTML exported: {html_path}")


def main():
    """Main demo function"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║              American Express - GenAI Offer Generator                       ║
    ║              Personalized, Policy-Compliant Offer System                    ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check if API key is configured
    from config import Config
    
    if not Config.OPENROUTER_API_KEY or Config.OPENROUTER_API_KEY == 'your_api_key_here':
        print("\n⚠️  WARNING: OPENROUTER_API_KEY not configured!")
        print("   Please edit config.py and update the OPENROUTER_API_KEY value.")
        print("   Get your API key from: https://openrouter.ai/\n")
        
        response = input("Continue with demo anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    try:
        # Initialize the system
        print("\n🚀 Initializing GenAI Reasoner System...")
        reasoner = GenAIReasoner(initialize_policies=True)
        
        # Demo 1: Policy System
        demo_policy_system(reasoner)
        
        input("\nPress Enter to continue to user demos...")
        
        # Demo 2: Single User
        response = demo_single_user(reasoner, 'USR001')
        
        if response:
            # Export results
            export_results(response, 'USR001')
        
        input("\nPress Enter to process all users...")
        
        # Demo 3: All Users
        demo_all_users(reasoner)
        
        print_section("DEMO COMPLETE")
        print("✅ All demonstrations completed successfully!")
        print("\n📁 Check the './output' directory for exported results")
        print("   - JSON files contain complete offer data")
        print("   - HTML files contain email-ready offer templates")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
