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
                'id': 'electronics_offer_policy',
                'content': '''Electronics and Technology Offers:
                - Eligible merchants: Electronics retailers, tech brands, online tech stores
                - Maximum discount: 15% back or 4x points
                - Partner brands: Apple, Best Buy, Microsoft, Samsung, Dell, and other authorized retailers
                - Extended warranty: Additional 1-2 year warranty on purchases
                - Purchase protection: Coverage for theft and accidental damage (90-120 days)
                - Minimum purchase: Typically $100-$200
                - High-ticket items: Computers, smartphones, tablets, TVs, cameras
                - Exclusions: Used/refurbished items may have different terms
                - Annual limits: Maximum $500 in electronics-specific statement credits per year''',
                'metadata': {'type': 'offer', 'domain': 'Electronics'}
            },
            {
                'id': 'business_services_offer_policy',
                'content': '''Business Services Category Policy:
                - Eligible services: Office supplies, software subscriptions, cloud services, professional services
                - Partner providers: Microsoft 365, Adobe, AWS, Google Workspace, FedEx, UPS
                - Maximum multiplier: Up to 5x points on business services
                - Software subscriptions: 3x-4x points on SaaS and business tools
                - Shipping services: 3x points at FedEx, UPS, USPS
                - Office supplies: 2x-3x points at Staples, Office Depot, Amazon Business
                - Minimum purchase: Typically $50
                - Bundled offers: Enhanced rewards for multiple business service categories
                - Annual caps: $50,000 in combined business services purchases per year
                - Small business bonus: Additional benefits for registered small business card members''',
                'metadata': {'type': 'offer', 'domain': 'Business Services'}
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
        
        # Add merchant data
        self._initialize_merchant_data()
    
    def _initialize_merchant_data(self):
        """Initialize merchant partnership data for personalized offers"""
        
        merchant_data = [
            {
                'id': 'travel_merchants',
                'content': '''Travel Partner Merchants:
                Airlines: Delta Air Lines (MERCH_DELTA_001), United Airlines (MERCH_UNITED_002), American Airlines (MERCH_AA_003), Southwest Airlines (MERCH_SW_004), JetBlue Airways (MERCH_JB_005)
                Hotels: Marriott Hotels (MERCH_MARRIOTT_101), Hilton Hotels (MERCH_HILTON_102), Hyatt Hotels (MERCH_HYATT_103), InterContinental Hotels (MERCH_IHG_104), Four Seasons (MERCH_4S_105)
                Car Rentals: Hertz (MERCH_HERTZ_201), Enterprise (MERCH_ENT_202), Avis (MERCH_AVIS_203), Budget (MERCH_BUDGET_204), National (MERCH_NAT_205)
                Travel Agencies: American Express Travel (MERCH_AMEXTRAVEL_301), Expedia (MERCH_EXPEDIA_302), Booking.com (MERCH_BOOKING_303)
                All merchants offer enhanced rewards and partner benefits for cardholders.''',
                'metadata': {'type': 'merchants', 'domain': 'Travel', 'category': 'partner_list'}
            },
            {
                'id': 'dining_merchants',
                'content': '''Dining Partner Merchants:
                Fine Dining: The Capital Grille (MERCH_CAPGRILLE_401), Ruth's Chris Steak House (MERCH_RUTHS_402), Morton's The Steakhouse (MERCH_MORTONS_403), Wolfgang Puck Restaurants (MERCH_WPUCK_404)
                Casual Dining: Cheesecake Factory (MERCH_CHEESECAKE_501), P.F. Chang's (MERCH_PFCHANG_502), Maggiano's (MERCH_MAGGIANO_503), Olive Garden (MERCH_OLIVE_504)
                Fast Casual: Shake Shack (MERCH_SHAKE_601), Chipotle (MERCH_CHIPOTLE_602), Panera Bread (MERCH_PANERA_603), Sweetgreen (MERCH_SWEET_604)
                Reservations: Resy Partner Restaurants (MERCH_RESY_701), OpenTable Network (MERCH_OPENTABLE_702)
                Coffee: Starbucks (MERCH_STARBUCKS_801), Dunkin' (MERCH_DUNKIN_802), Peet's Coffee (MERCH_PEETS_803)
                Enhanced rewards available at all participating dining partners.''',
                'metadata': {'type': 'merchants', 'domain': 'Dining', 'category': 'partner_list'}
            },
            {
                'id': 'electronics_merchants',
                'content': '''Electronics Partner Merchants:
                Major Retailers: Best Buy (MERCH_BESTBUY_901), B&H Photo (MERCH_BH_902), Micro Center (MERCH_MICRO_903), Newegg (MERCH_NEWEGG_904), Adorama (MERCH_ADORAMA_905)
                Brand Stores: Apple Store (MERCH_APPLE_1001), Microsoft Store (MERCH_MSFT_1002), Samsung (MERCH_SAMSUNG_1003), Dell (MERCH_DELL_1004), HP Store (MERCH_HP_1005)
                Online Marketplaces: Amazon Electronics (MERCH_AMAZON_ELEC_1101), Walmart Electronics (MERCH_WALMART_ELEC_1102), Target Electronics (MERCH_TARGET_ELEC_1103)
                Specialty: GameStop (MERCH_GAMESTOP_1201), Crutchfield (MERCH_CRUTCH_1202), Verizon (MERCH_VZW_1203), AT&T (MERCH_ATT_1204)
                All merchants offer purchase protection, extended warranty, and enhanced rewards.''',
                'metadata': {'type': 'merchants', 'domain': 'Electronics', 'category': 'partner_list'}
            },
            {
                'id': 'retail_merchants',
                'content': '''Retail Partner Merchants:
                Department Stores: Macy's (MERCH_MACYS_1301), Nordstrom (MERCH_NORD_1302), Bloomingdale's (MERCH_BLOOM_1303), Saks Fifth Avenue (MERCH_SAKS_1304), Neiman Marcus (MERCH_NEIMAN_1305)
                Fashion: Nike (MERCH_NIKE_1401), Adidas (MERCH_ADIDAS_1402), Lululemon (MERCH_LULU_1403), Gap (MERCH_GAP_1404), Zara (MERCH_ZARA_1405)
                Home: Home Depot (MERCH_HOMEDEPOT_1501), Lowe's (MERCH_LOWES_1502), Bed Bath & Beyond (MERCH_BBB_1503), Williams Sonoma (MERCH_WS_1504)
                Online: Amazon (MERCH_AMAZON_1601), Walmart (MERCH_WALMART_1602), Target (MERCH_TARGET_1603), Costco (MERCH_COSTCO_1604)
                Luxury: Tiffany & Co. (MERCH_TIFFANY_1701), Gucci (MERCH_GUCCI_1702), Louis Vuitton (MERCH_LV_1703)
                Special rates and exclusive access at partner retailers.''',
                'metadata': {'type': 'merchants', 'domain': 'Retail', 'category': 'partner_list'}
            },
            {
                'id': 'business_services_merchants',
                'content': '''Business Services Partner Merchants:
                Cloud Services: Amazon Web Services (MERCH_AWS_1801), Microsoft Azure (MERCH_AZURE_1802), Google Cloud Platform (MERCH_GCP_1803), Salesforce (MERCH_SFDC_1804)
                Software: Microsoft 365 (MERCH_M365_1901), Adobe Creative Cloud (MERCH_ADOBE_1902), Zoom (MERCH_ZOOM_1903), Slack (MERCH_SLACK_1904), DocuSign (MERCH_DOCU_1905)
                Shipping: FedEx (MERCH_FEDEX_2001), UPS (MERCH_UPS_2002), DHL (MERCH_DHL_2003), USPS (MERCH_USPS_2004)
                Office Supplies: Staples (MERCH_STAPLES_2101), Office Depot (MERCH_OFFICEDEPOT_2102), Amazon Business (MERCH_AMZBIZ_2103), Quill (MERCH_QUILL_2104)
                Professional Services: LinkedIn Premium (MERCH_LINKEDIN_2201), Indeed (MERCH_INDEED_2202), QuickBooks (MERCH_QB_2203)
                Enhanced rewards and business benefits at all partner locations.''',
                'metadata': {'type': 'merchants', 'domain': 'Business Services', 'category': 'partner_list'}
            },
            {
                'id': 'entertainment_merchants',
                'content': '''Entertainment Partner Merchants:
                Streaming: Netflix (MERCH_NETFLIX_2301), Disney+ (MERCH_DISNEY_2302), Hulu (MERCH_HULU_2303), Spotify (MERCH_SPOTIFY_2304), Apple Music (MERCH_APPLEMUSIC_2305), YouTube Premium (MERCH_YT_2306)
                Ticketing: Ticketmaster (MERCH_TM_2401), Live Nation (MERCH_LIVENATION_2402), StubHub (MERCH_STUBHUB_2403), SeatGeek (MERCH_SEATGEEK_2404)
                Movies: AMC Theatres (MERCH_AMC_2501), Regal Cinemas (MERCH_REGAL_2502), Cinemark (MERCH_CINEMARK_2503), Fandango (MERCH_FANDANGO_2504)
                Gaming: PlayStation Network (MERCH_PSN_2601), Xbox Live (MERCH_XBOX_2602), Nintendo eShop (MERCH_NINTENDO_2603), Steam (MERCH_STEAM_2604)
                Events: Madison Square Garden (MERCH_MSG_2701), Live Nation Venues (MERCH_LN_VENUES_2702)
                Exclusive presales, bonus points, and credits available at partner entertainment merchants.''',
                'metadata': {'type': 'merchants', 'domain': 'Entertainment', 'category': 'partner_list'}
            }
        ]
        
        self.add_policies_bulk(merchant_data)
        print(f"✓ Initialized {len(merchant_data)} merchant partner datasets")
