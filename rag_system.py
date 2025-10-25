"""
RAG System for Policy and Compliance Rule Retrieval
Handles document ingestion, embedding, and retrieval using ChromaDB
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import os
from config import Config


class PolicyRAG:
    """Retrieval-Augmented Generation system for policy documents"""
    
    def __init__(self, collection_name: str = "policies"):
        """
        Initialize the RAG system with ChromaDB
        
        Args:
            collection_name: Name of the collection to store policies
        """
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=Config.VECTOR_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Policy and compliance documents"}
        )
    
    def add_policy(self, policy_id: str, content: str, metadata: Dict[str, Any] = None):
        """
        Add a policy document to the vector store
        
        Args:
            policy_id: Unique identifier for the policy
            content: Text content of the policy
            metadata: Additional metadata (type, domain, etc.)
        """
        embedding = self.embedding_model.encode(content).tolist()
        
        self.collection.add(
            ids=[policy_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata or {}]
        )
    
    def add_policies_bulk(self, policies: List[Dict[str, Any]]):
        """
        Add multiple policies in bulk
        
        Args:
            policies: List of dicts with 'id', 'content', and optional 'metadata'
        """
        ids = [p['id'] for p in policies]
        contents = [p['content'] for p in policies]
        metadatas = [p.get('metadata', {}) for p in policies]
        
        embeddings = self.embedding_model.encode(contents).tolist()
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas
        )
    
    def retrieve_relevant_policies(
        self, 
        query: str, 
        domain: str = None, 
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant policies based on query and optional domain filter
        
        Args:
            query: Search query (user context or offer context)
            domain: Optional domain filter (Travel, Dining, etc.)
            n_results: Number of results to return
            
        Returns:
            List of relevant policy documents with metadata
        """
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Build where filter if domain is specified
        where_filter = {"domain": domain} if domain else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter
        )
        
        # Format results
        policies = []
        if results['documents'] and len(results['documents']) > 0:
            for i, doc in enumerate(results['documents'][0]):
                policies.append({
                    'id': results['ids'][0][i],
                    'content': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
        
        return policies
    
    def get_all_policies(self) -> List[Dict[str, Any]]:
        """Get all policies from the collection"""
        results = self.collection.get()
        
        policies = []
        if results['documents']:
            for i, doc in enumerate(results['documents']):
                policies.append({
                    'id': results['ids'][i],
                    'content': doc,
                    'metadata': results['metadatas'][i] if results['metadatas'] else {}
                })
        
        return policies
    
    def clear_policies(self):
        """Clear all policies from the collection"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"description": "Policy and compliance documents"}
        )
    
    def initialize_default_policies(self, force_reload: bool = False):
        """
        Initialize the system with default policies
        
        Args:
            force_reload: If True, clears and reloads policies. If False, only loads if empty.
        """
        # Check if policies already exist
        existing_policies = self.get_all_policies()
        
        if not force_reload and len(existing_policies) > 0:
            print(f"✓ Found {len(existing_policies)} existing policies in database - skipping initialization")
            return
        
        print("Loading default policies into vector database...")
        
        default_policies = [
            {
                'id': 'points_multiplier_policy',
                'content': '''Points Multiplier Policy:
                - Standard earn rate: 1x Membership Rewards points per dollar
                - Maximum bonus multiplier: 5x points in any category
                - Promotional periods: Maximum 3 months duration
                - No more than 3 active multiplier offers per card member per month
                - Minimum spend requirement: $50 for multiplier activation
                - Excludes cash advances, fees, and balance transfers''',
                'metadata': {'type': 'rewards', 'domain': 'all'}
            },
            {
                'id': 'travel_offer_policy',
                'content': '''Travel Offer Guidelines:
                - Partner airlines: Must be from approved airline partner list
                - Booking requirement: Direct booking through American Express Travel or partner sites
                - Blackout dates: Major holidays and peak seasons may be excluded
                - Point caps: Maximum 50,000 bonus points per travel offer
                - Eligible categories: Flights, hotels, car rentals, vacation packages
                - Terms: Offers valid for 30-60 days from issue date
                - Elite status: Cardholders with higher tier may receive enhanced offers''',
                'metadata': {'type': 'offer', 'domain': 'Travel'}
            },
            {
                'id': 'dining_offer_policy',
                'content': '''Dining Rewards Policy:
                - Participating merchants: Restaurants enrolled in American Express dining program
                - Standard multiplier: Up to 4x points at restaurants
                - Monthly caps: $25,000 in combined purchases per month
                - Excluded categories: Fast food and quick service may have lower rates
                - Reservation bonus: Additional points for reservations made through Resy
                - Small business focus: Enhanced rewards at locally-owned restaurants''',
                'metadata': {'type': 'offer', 'domain': 'Dining'}
            },
            {
                'id': 'retail_offer_policy',
                'content': '''Retail Shopping Offers:
                - Eligible merchants: Department stores, online retailers, specialty shops
                - Maximum discount: 20% back or 5x points
                - Merchant partnerships: Offers rotate quarterly with partner retailers
                - Online shopping: Enhanced offers through Amex Offers portal
                - Minimum purchase: Typically $25-$50
                - Exclusions: Gift card purchases, prior purchases, returns
                - Small Business Saturday: Special 2x-3x points at small businesses''',
                'metadata': {'type': 'offer', 'domain': 'Retail'}
            },
            {
                'id': 'entertainment_offer_policy',
                'content': '''Entertainment Category Policy:
                - Eligible purchases: Concerts, sporting events, movies, streaming services
                - Partner venues: Live Nation, Ticketmaster, select theaters
                - Streaming multiplier: Up to 3x points on streaming services
                - Presale access: Early ticket access for select events
                - Statement credits: Up to $20/month for streaming or entertainment
                - Annual limits: Maximum $300 in entertainment-specific credits per year''',
                'metadata': {'type': 'offer', 'domain': 'Entertainment'}
            },
            {
                'id': 'compliance_policy',
                'content': '''Compliance and Regulatory Requirements:
                - Privacy: Must comply with GLBA, CCPA, and GDPR where applicable
                - Fair lending: Offers must be non-discriminatory and comply with ECOA
                - Truth in lending: All terms must be clearly disclosed
                - Anti-money laundering: Monitor for suspicious patterns
                - Marketing consent: Must have opt-in for promotional communications
                - Data retention: User data stored according to retention policies
                - Accessibility: All offers must be accessible per ADA requirements
                - Brand guidelines: Communications must follow American Express brand standards''',
                'metadata': {'type': 'compliance', 'domain': 'all'}
            },
            {
                'id': 'frequency_policy',
                'content': '''Offer Frequency and Timing Policy:
                - Maximum offers per card member: 3 personalized offers per month
                - Minimum gap between offers: 7 days for same category
                - Cooling period: 30 days after declined offer in same category
                - High-value offers: Limited to once per quarter per domain
                - Reactivation offers: Special handling for dormant accounts
                - Seasonal timing: Adjust for holidays, travel seasons, and merchant patterns
                - Response time: Users have 7-30 days to activate offers
                - Expiration: Activated offers valid for 30-90 days based on category''',
                'metadata': {'type': 'business_rules', 'domain': 'all'}
            },
            {
                'id': 'eligibility_policy',
                'content': '''Card Member Eligibility Requirements:
                - Account status: Must be current (no delinquencies over 30 days)
                - Account age: Minimum 90 days for premium offers
                - Credit limit: Offers scaled to available credit and spending power
                - Previous offers: History of offer acceptance increases eligibility
                - Opt-in status: Must be opted in to marketing communications
                - Geographic restrictions: Some offers limited by region or country
                - Product type: Certain offers exclusive to premium card products
                - Spending threshold: Minimum $500/month average for targeted offers''',
                'metadata': {'type': 'eligibility', 'domain': 'all'}
            }
        ]
        
        # Clear existing only if force_reload
        if force_reload:
            print("Force reload enabled - clearing existing policies...")
            try:
                self.clear_policies()
            except:
                pass
        
        self.add_policies_bulk(default_policies)
        print(f"✓ Initialized {len(default_policies)} default policies")
