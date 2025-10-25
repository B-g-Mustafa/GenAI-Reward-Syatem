"""
User Data Store

Simulates a database/data warehouse that stores user information.
In production, this would connect to actual data sources.
"""

from typing import Optional, Dict, Any
from sample_data import generate_sample_user_data


class UserDataStore:
    """
    Centralized user data storage and retrieval
    
    In production, this would:
    - Connect to customer database
    - Query data warehouse
    - Fetch from cache (Redis)
    - Aggregate from multiple sources
    """
    
    def __init__(self):
        """Initialize with sample data (for demo purposes)"""
        self._users = {}
        self._load_sample_data()
    
    def _load_sample_data(self):
        """Load sample user data"""
        sample_users = generate_sample_user_data()
        for user in sample_users:
            self._users[user['user_id']] = user
        print(f"✅ Loaded {len(self._users)} users into data store")
    
    def get_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve complete user data by user_id
        
        Args:
            user_id: User identifier
            
        Returns:
            Complete user data dict or None if not found
        """
        return self._users.get(user_id)
    
    def add_user(self, user_data: Dict[str, Any]) -> bool:
        """
        Add or update user data
        
        Args:
            user_data: Complete user data dict with user_id
            
        Returns:
            True if successful
        """
        if 'user_id' not in user_data:
            raise ValueError("user_data must contain 'user_id'")
        
        self._users[user_data['user_id']] = user_data
        return True
    
    def user_exists(self, user_id: str) -> bool:
        """Check if user exists in store"""
        return user_id in self._users
    
    def get_all_user_ids(self):
        """Get list of all user IDs"""
        return list(self._users.keys())
    
    def get_user_count(self) -> int:
        """Get total number of users"""
        return len(self._users)


# Global instance (singleton pattern)
_user_store_instance = None


def get_user_store() -> UserDataStore:
    """Get global user data store instance"""
    global _user_store_instance
    if _user_store_instance is None:
        _user_store_instance = UserDataStore()
    return _user_store_instance
