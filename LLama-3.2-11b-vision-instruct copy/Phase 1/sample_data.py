"""
Sample user data and ML recommendations for testing
"""

from datetime import datetime, timedelta
import random


def generate_sample_user_data():
    """Generate sample user data for testing"""
    
    # User 1: Travel enthusiast with decreasing spend
    user1 = {
        'user_id': 'USR001',
        'profile': {
            'name': 'Alex Johnson',
            'segment': 'Premium',
            'tenure_months': 36,
            'demographics': {
                'age_group': '35-44',
                'location': 'New York, NY'
            }
        },
        'transaction_history': [
            # Recent travel transactions (last 60 days) - decreased
            {'date': (datetime.now() - timedelta(days=10)).isoformat(), 'merchant': 'Delta Airlines', 'amount': 450.00, 'category': 'Travel'},
            {'date': (datetime.now() - timedelta(days=25)).isoformat(), 'merchant': 'Marriott Hotels', 'amount': 380.00, 'category': 'Travel'},
            
            # Previous period (60-120 days ago) - higher spend
            {'date': (datetime.now() - timedelta(days=70)).isoformat(), 'merchant': 'United Airlines', 'amount': 1200.00, 'category': 'Travel'},
            {'date': (datetime.now() - timedelta(days=75)).isoformat(), 'merchant': 'Hilton Hotels', 'amount': 680.00, 'category': 'Travel'},
            {'date': (datetime.now() - timedelta(days=85)).isoformat(), 'merchant': 'Enterprise Rent-A-Car', 'amount': 320.00, 'category': 'Travel'},
            {'date': (datetime.now() - timedelta(days=90)).isoformat(), 'merchant': 'Delta Airlines', 'amount': 890.00, 'category': 'Travel'},
            {'date': (datetime.now() - timedelta(days=105)).isoformat(), 'merchant': 'Airbnb', 'amount': 540.00, 'category': 'Travel'},
            
            # Other categories
            {'date': (datetime.now() - timedelta(days=5)).isoformat(), 'merchant': 'Whole Foods', 'amount': 120.00, 'category': 'Groceries'},
            {'date': (datetime.now() - timedelta(days=15)).isoformat(), 'merchant': 'Shell Gas Station', 'amount': 55.00, 'category': 'Gas'},
            {'date': (datetime.now() - timedelta(days=20)).isoformat(), 'merchant': 'The Capital Grille', 'amount': 180.00, 'category': 'Dining'},
        ],
        'engagement_behavior': {
            'app_opens': 25,
            'offer_clicks': 8,
            'offers_shown': 12,
            'redemptions': 5,
            'offers_accepted': 6,
            'last_login': (datetime.now() - timedelta(days=2)).isoformat()
        },
        'offer_history': [
            {'domain': 'Travel', 'status': 'accepted', 'date': (datetime.now() - timedelta(days=90)).isoformat()},
            {'domain': 'Travel', 'status': 'accepted', 'date': (datetime.now() - timedelta(days=120)).isoformat()},
            {'domain': 'Dining', 'status': 'accepted', 'date': (datetime.now() - timedelta(days=60)).isoformat()},
            {'domain': 'Retail', 'status': 'ignored', 'date': (datetime.now() - timedelta(days=45)).isoformat()},
            {'domain': 'Travel', 'status': 'accepted', 'date': (datetime.now() - timedelta(days=150)).isoformat()},
            {'domain': 'Entertainment', 'status': 'ignored', 'date': (datetime.now() - timedelta(days=30)).isoformat()},
        ]
    }
    
    # User 2: New to dining category
    user2 = {
        'user_id': 'USR002',
        'profile': {
            'name': 'Sarah Chen',
            'segment': 'Standard',
            'tenure_months': 18,
            'demographics': {
                'age_group': '25-34',
                'location': 'San Francisco, CA'
            }
        },
        'transaction_history': [
            # Recent diverse spending
            {'date': (datetime.now() - timedelta(days=3)).isoformat(), 'merchant': 'Amazon', 'amount': 89.99, 'category': 'Retail'},
            {'date': (datetime.now() - timedelta(days=7)).isoformat(), 'merchant': 'Target', 'amount': 145.50, 'category': 'Retail'},
            {'date': (datetime.now() - timedelta(days=12)).isoformat(), 'merchant': 'Spotify', 'amount': 15.99, 'category': 'Entertainment'},
            {'date': (datetime.now() - timedelta(days=18)).isoformat(), 'merchant': 'Chipotle', 'amount': 24.50, 'category': 'Dining'},
            {'date': (datetime.now() - timedelta(days=25)).isoformat(), 'merchant': 'Starbucks', 'amount': 12.75, 'category': 'Dining'},
            {'date': (datetime.now() - timedelta(days=30)).isoformat(), 'merchant': 'Uber Eats', 'amount': 35.20, 'category': 'Dining'},
            
            # Very little dining in previous period
            {'date': (datetime.now() - timedelta(days=80)).isoformat(), 'merchant': 'Best Buy', 'amount': 299.00, 'category': 'Retail'},
            {'date': (datetime.now() - timedelta(days=95)).isoformat(), 'merchant': 'Apple Store', 'amount': 1299.00, 'category': 'Retail'},
        ],
        'engagement_behavior': {
            'app_opens': 12,
            'offer_clicks': 3,
            'offers_shown': 8,
            'redemptions': 2,
            'offers_accepted': 3,
            'last_login': (datetime.now() - timedelta(days=5)).isoformat()
        },
        'offer_history': [
            {'domain': 'Retail', 'status': 'accepted', 'date': (datetime.now() - timedelta(days=60)).isoformat()},
            {'domain': 'Retail', 'status': 'accepted', 'date': (datetime.now() - timedelta(days=120)).isoformat()},
            {'domain': 'Gas', 'status': 'ignored', 'date': (datetime.now() - timedelta(days=40)).isoformat()},
        ]
    }
    
    # User 3: High spender with increasing entertainment spend
    user3 = {
        'user_id': 'USR003',
        'profile': {
            'name': 'Michael Torres',
            'segment': 'Platinum',
            'tenure_months': 72,
            'demographics': {
                'age_group': '45-54',
                'location': 'Los Angeles, CA'
            }
        },
        'transaction_history': [
            # Recent entertainment surge
            {'date': (datetime.now() - timedelta(days=5)).isoformat(), 'merchant': 'Ticketmaster', 'amount': 450.00, 'category': 'Entertainment'},
            {'date': (datetime.now() - timedelta(days=12)).isoformat(), 'merchant': 'AMC Theatres', 'amount': 85.00, 'category': 'Entertainment'},
            {'date': (datetime.now() - timedelta(days=18)).isoformat(), 'merchant': 'Live Nation', 'amount': 320.00, 'category': 'Entertainment'},
            {'date': (datetime.now() - timedelta(days=25)).isoformat(), 'merchant': 'Netflix', 'amount': 19.99, 'category': 'Entertainment'},
            {'date': (datetime.now() - timedelta(days=35)).isoformat(), 'merchant': 'Spotify Premium', 'amount': 15.99, 'category': 'Entertainment'},
            {'date': (datetime.now() - timedelta(days=45)).isoformat(), 'merchant': 'Broadway Show', 'amount': 380.00, 'category': 'Entertainment'},
            
            # Previous period - less entertainment
            {'date': (datetime.now() - timedelta(days=75)).isoformat(), 'merchant': 'AMC Theatres', 'amount': 45.00, 'category': 'Entertainment'},
            {'date': (datetime.now() - timedelta(days=100)).isoformat(), 'merchant': 'Spotify Premium', 'amount': 15.99, 'category': 'Entertainment'},
            
            # High dining spend throughout
            {'date': (datetime.now() - timedelta(days=8)).isoformat(), 'merchant': 'Nobu', 'amount': 285.00, 'category': 'Dining'},
            {'date': (datetime.now() - timedelta(days=22)).isoformat(), 'merchant': 'Cut by Wolfgang Puck', 'amount': 340.00, 'category': 'Dining'},
        ],
        'engagement_behavior': {
            'app_opens': 35,
            'offer_clicks': 15,
            'offers_shown': 18,
            'redemptions': 12,
            'offers_accepted': 14,
            'last_login': (datetime.now() - timedelta(days=1)).isoformat()
        },
        'offer_history': [
            {'domain': 'Entertainment', 'status': 'accepted', 'date': (datetime.now() - timedelta(days=60)).isoformat()},
            {'domain': 'Dining', 'status': 'accepted', 'date': (datetime.now() - timedelta(days=45)).isoformat()},
            {'domain': 'Travel', 'status': 'accepted', 'date': (datetime.now() - timedelta(days=90)).isoformat()},
            {'domain': 'Entertainment', 'status': 'accepted', 'date': (datetime.now() - timedelta(days=120)).isoformat()},
            {'domain': 'Dining', 'status': 'accepted', 'date': (datetime.now() - timedelta(days=30)).isoformat()},
            {'domain': 'Retail', 'status': 'accepted', 'date': (datetime.now() - timedelta(days=75)).isoformat()},
        ]
    }
    
    return [user1, user2, user3]


def generate_sample_ml_recommendations():
    """Generate sample ML model outputs"""
    
    recommendations = [
        {
            'user_id': 'USR001',
            'offer_flag': True,
            'domain': 'Travel',
            'confidence_score': 0.82
        },
        {
            'user_id': 'USR002',
            'offer_flag': True,
            'domain': 'Dining',
            'confidence_score': 0.68
        },
        {
            'user_id': 'USR003',
            'offer_flag': True,
            'domain': 'Entertainment',
            'confidence_score': 0.91
        }
    ]
    
    return recommendations


# Create a mapping for easy access
SAMPLE_USERS = {
    'USR001': None,  # Will be populated when function is called
    'USR002': None,
    'USR003': None
}

SAMPLE_ML_RECOMMENDATIONS = {
    'USR001': None,
    'USR002': None,
    'USR003': None
}


def get_sample_data(user_id: str = None):
    """
    Get sample data for testing
    
    Args:
        user_id: Specific user ID to retrieve, or None for all users
        
    Returns:
        Tuple of (user_data, ml_recommendation) or list of all samples
    """
    users = generate_sample_user_data()
    recommendations = generate_sample_ml_recommendations()
    
    # Update mappings
    for user in users:
        SAMPLE_USERS[user['user_id']] = user
    
    for rec in recommendations:
        SAMPLE_ML_RECOMMENDATIONS[rec['user_id']] = rec
    
    if user_id:
        return SAMPLE_USERS.get(user_id), SAMPLE_ML_RECOMMENDATIONS.get(user_id)
    
    return list(zip(users, recommendations))
