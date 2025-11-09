"""
User Data Store

Connects to PostgreSQL database to retrieve user information.
"""

from typing import Optional, Dict, Any
from decimal import Decimal
import psycopg2
import psycopg2.extras
import json


class UserDataStore:
    """
    PostgreSQL-backed user data storage and retrieval
    
    Connects to Neon PostgreSQL database to fetch user profiles,
    transaction history, and behavioral data.
    """
    
    # Database connection parameters
    DB_CONFIG = {
        'host': 'ep-orange-band-a4bod0rg.us-east-1.aws.neon.tech',
        'database': 'neondb',
        'user': 'neondb_owner',
        'password': 'npg_0qI2wkoAdUcf',
        'sslmode': 'require'
    }
    
    def __init__(self):
        """Initialize PostgreSQL connection"""
        self.conn = None
        self._connect()
    
    def _connect(self):
        """Establish connection to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(**self.DB_CONFIG)
            print(f"✅ Connected to PostgreSQL database at {self.DB_CONFIG['host']}")
        except Exception as e:
            print(f"❌ Failed to connect to PostgreSQL: {e}")
            raise
    
    def _ensure_connection(self):
        """Ensure database connection is active, reconnect if needed"""
        try:
            if self.conn is None or self.conn.closed:
                self._connect()
            else:
                # Test connection
                cursor = self.conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
        except Exception:
            self._connect()
    
    def _convert_decimals(self, obj: Any) -> Any:
        """
        Recursively convert Decimal objects to float for JSON compatibility
        
        Args:
            obj: Object to convert (dict, list, or any value)
            
        Returns:
            Object with Decimals converted to float
        """
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_decimals(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_decimals(item) for item in obj]
        else:
            return obj
    
    def get_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve complete user data by user_id from PostgreSQL
        
        Args:
            user_id: User identifier
            
        Returns:
            Complete user data dict or None if not found
        """
        try:
            self._ensure_connection()
            
            cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Query user data from amex_data table
            query = """
                SELECT * FROM amex_data 
                WHERE user_id = %s
            """
            
            cursor.execute(query, (user_id,))
            user_row = cursor.fetchone()
            cursor.close()
            
            if user_row is None:
                return None
            
            # Convert RealDictRow to regular dict
            user_data = dict(user_row)
            
            # Parse JSON fields if they're stored as strings (for backward compatibility)
            # Now expecting flat structure, but handle nested if present
            json_fields = ['transaction_history', 'spending_by_category', 
                          'engagement_metrics', 'offer_history', 'profile']
            for field in json_fields:
                if field in user_data and isinstance(user_data[field], str):
                    try:
                        user_data[field] = json.loads(user_data[field])
                    except:
                        pass
            
            # Handle both flat and nested structures for backward compatibility
            # If we have a nested 'profile', flatten it to top level
            if 'profile' in user_data and isinstance(user_data['profile'], dict):
                profile_data = user_data.pop('profile')
                # Only add profile fields if they don't already exist at top level
                for key, value in profile_data.items():
                    if key not in user_data:
                        user_data[key] = value
            
            # Convert all Decimal types to float for JSON compatibility
            user_data = self._convert_decimals(user_data)
            
            return user_data
            
        except Exception as e:
            print(f"❌ Error retrieving user data for {user_id}: {e}")
            return None
    
    def add_user(self, user_data: Dict[str, Any]) -> bool:
        """
        Add or update user data in PostgreSQL
        
        Args:
            user_data: Complete user data dict with user_id
            
        Returns:
            True if successful
        """
        if 'user_id' not in user_data:
            raise ValueError("user_data must contain 'user_id'")
        
        try:
            self._ensure_connection()
            
            cursor = self.conn.cursor()
            
            # Note: This is a simplified upsert example
            # For amex_data table with flat columns, you'd need to specify all columns
            # This method may need customization based on your insert requirements
            query = """
                INSERT INTO amex_data (user_id, data) 
                VALUES (%s, %s)
                ON CONFLICT (user_id) 
                DO UPDATE SET data = EXCLUDED.data
            """
            
            cursor.execute(query, (user_data['user_id'], json.dumps(user_data)))
            self.conn.commit()
            cursor.close()
            
            return True
            
        except Exception as e:
            print(f"❌ Error adding user data: {e}")
            if self.conn:
                self.conn.rollback()
            return False
    
    def user_exists(self, user_id: str) -> bool:
        """Check if user exists in PostgreSQL"""
        try:
            self._ensure_connection()
            
            cursor = self.conn.cursor()
            query = "SELECT 1 FROM amex_data WHERE user_id = %s LIMIT 1"
            cursor.execute(query, (user_id,))
            exists = cursor.fetchone() is not None
            cursor.close()
            
            return exists
            
        except Exception as e:
            print(f"❌ Error checking user existence: {e}")
            return False
    
    def get_all_user_ids(self):
        """Get list of all user IDs from PostgreSQL"""
        try:
            self._ensure_connection()
            
            cursor = self.conn.cursor()
            query = "SELECT user_id FROM amex_data"
            cursor.execute(query)
            user_ids = [row[0] for row in cursor.fetchall()]
            cursor.close()
            
            return user_ids
            
        except Exception as e:
            print(f"❌ Error retrieving user IDs: {e}")
            return []
    
    def get_user_count(self) -> int:
        """Get total number of users in PostgreSQL"""
        try:
            self._ensure_connection()
            
            cursor = self.conn.cursor()
            query = "SELECT COUNT(*) FROM amex_data"
            cursor.execute(query)
            count = cursor.fetchone()[0]
            cursor.close()
            
            return count
            
        except Exception as e:
            print(f"❌ Error getting user count: {e}")
            return 0
    
    def __del__(self):
        """Close database connection on cleanup"""
        if self.conn and not self.conn.closed:
            self.conn.close()


# Global instance (singleton pattern)
_user_store_instance = None


def get_user_store() -> UserDataStore:
    """Get global user data store instance"""
    global _user_store_instance
    if _user_store_instance is None:
        _user_store_instance = UserDataStore()
    return _user_store_instance
