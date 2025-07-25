import asyncio
import time
import logging
import re
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup
import redis
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from serpapi import GoogleSearch

from .schemas import (
    SearchRequest, SearchResponse, SearchResult,
    CrawlRequest, CrawlResponse, CrawledPage,
    DomainReputation, LinkPrioritization, DiscoveryStats,
    TimeFilter
)
from app.core.config import settings
from app.modules.module1.services import keyword_extractor

# Configure logging
logger = logging.getLogger(__name__)

# Database models
Base = declarative_base()

class SearchResultDB(Base):
    __tablename__ = "search_results"
    
    id = Column(Integer, primary_key=True)
    url = Column(String(500), unique=True, nullable=False)
    title = Column(String(500), nullable=False)
    snippet = Column(Text, nullable=False)
    domain = Column(String(100), nullable=False)
    relevance_score = Column(Float, nullable=False)
    estimated_date = Column(DateTime, nullable=True)
    search_query = Column(String(500), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class CrawledPageDB(Base):
    """Database model for crawled pages"""
    __tablename__ = "crawled_pages"
    
    id = Column(Integer, primary_key=True)
    url = Column(String(500), unique=True, nullable=False)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    html_content = Column(Text, nullable=False)
    domain = Column(String(100), nullable=False)
    crawl_depth = Column(Integer, nullable=False)
    relevance_score = Column(Float, nullable=False)
    discovered_links = Column(Text, nullable=False)  # JSON string
    meta_data = Column(Text, nullable=False)  # JSON string
    crawl_timestamp = Column(DateTime, default=datetime.utcnow)
    response_time = Column(Float, nullable=False)
    status_code = Column(Integer, nullable=False)


class DomainReputationDB(Base):
    __tablename__ = "domain_reputations"
    
    id = Column(Integer, primary_key=True)
    domain = Column(String(100), unique=True, nullable=False)
    reputation_score = Column(Float, nullable=False)
    is_trusted = Column(Boolean, default=False)
    category = Column(String(100), nullable=False)
    last_checked = Column(DateTime, default=datetime.utcnow)


class DynamicSourceDiscovery:
    def __init__(self):
        self.initialized = False
        self.sentence_model = None
        self.redis_client = None
        self.db_session = None
        self.serpapi_key = settings.serpapi_key
        self.serpapi_enabled = bool(self.serpapi_key)
        
        # Debug logging
        logger.info(f"SerpApi key loaded: {self.serpapi_key[:10] if self.serpapi_key else 'None'}...")
        logger.info(f"SerpApi enabled: {self.serpapi_enabled}")
        if not self.serpapi_key:
            logger.error("❌ SerpApi key not found in settings. Check .env file.")
        elif self.serpapi_key == "your_serpapi_key_here" or self.serpapi_key == "e5d3f87b85d554019aa297603ed0da2f5b9198eb":
            logger.error("❌ SerpApi key is default/placeholder value. Please set real key in .env file.")
        else:
            logger.info("✅ SerpApi key is properly configured")
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
        
        # Trusted domains for reputation checking
        self.trusted_domains = {
            'arxiv.org': {'score': 0.9, 'category': 'academic'},
            'nature.com': {'score': 0.95, 'category': 'scientific'},
            'science.org': {'score': 0.95, 'category': 'scientific'},
            'ieee.org': {'score': 0.9, 'category': 'technical'},
            'acm.org': {'score': 0.9, 'category': 'technical'},
            'springer.com': {'score': 0.85, 'category': 'academic'},
            'tandfonline.com': {'score': 0.85, 'category': 'academic'},
            'sciencedirect.com': {'score': 0.9, 'category': 'scientific'},
            'researchgate.net': {'score': 0.8, 'category': 'academic'},
            'github.com': {'score': 0.8, 'category': 'technical'},
            'stackoverflow.com': {'score': 0.7, 'category': 'technical'},
            'medium.com': {'score': 0.6, 'category': 'blog'},
            'towardsdatascience.com': {'score': 0.7, 'category': 'blog'},
            'kdnuggets.com': {'score': 0.7, 'category': 'blog'},
            'techcrunch.com': {'score': 0.7, 'category': 'news'},
            'venturebeat.com': {'score': 0.7, 'category': 'news'},
            'wired.com': {'score': 0.8, 'category': 'news'},
            'mit.edu': {'score': 0.95, 'category': 'academic'},
            'stanford.edu': {'score': 0.95, 'category': 'academic'},
            'berkeley.edu': {'score': 0.95, 'category': 'academic'},
            'cmu.edu': {'score': 0.95, 'category': 'academic'},
            'ox.ac.uk': {'score': 0.95, 'category': 'academic'},
            'cam.ac.uk': {'score': 0.95, 'category': 'academic'},
            'ethz.ch': {'score': 0.95, 'category': 'academic'},
            'epfl.ch': {'score': 0.95, 'category': 'academic'}
        }
        
        # File extensions to skip
        self.skip_extensions = {
            '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
            '.zip', '.rar', '.tar', '.gz', '.7z',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.ico',
            '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm',
            '.mp3', '.wav', '.flac', '.aac',
            '.exe', '.msi', '.dmg', '.deb', '.rpm'
        }
        
        # Social media domains to skip
        self.social_media_domains = {
            'facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com',
            'youtube.com', 'reddit.com', 'pinterest.com', 'tumblr.com',
            'snapchat.com', 'tiktok.com', 'whatsapp.com', 'telegram.org'
        }
    
    async def initialize(self):
        """Initialize all components"""
        try:
            logger.info("Initializing Dynamic Source Discovery service...")
            
            # Initialize sentence transformer for semantic similarity (optional)
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Sentence transformer initialized")
            except Exception as e:
                logger.warning(f"Sentence transformer initialization failed: {e}")
                self.sentence_model = None
            
            # Initialize Redis for queue management (optional)
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
                # Test connection
                self.redis_client.ping()
                logger.info("Redis client initialized")
            except Exception as e:
                logger.warning(f"Redis initialization failed: {e}")
                self.redis_client = None
            
            # Initialize database (optional)
            try:
                engine = create_engine('sqlite:///discovery.db')
                Base.metadata.create_all(engine)
                Session = sessionmaker(bind=engine)
                self.db_session = Session()
                logger.info("Database initialized")
            except Exception as e:
                logger.warning(f"Database initialization failed: {e}")
                self.db_session = None
            
            self.initialized = True
            logger.info("Dynamic Source Discovery service initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Dynamic Source Discovery service: {e}")
            # Don't raise, just mark as not fully initialized
            self.initialized = False
    
    async def _extract_keywords_from_prompt(self, prompt: str, max_keywords: int = 20) -> List[str]:
        """
        Extract keywords from prompt using Module 1
        """
        try:
            logger.info(f"Extracting keywords from prompt using Module 1")
            
            # Use Module 1 to extract keywords
            keyword_result = keyword_extractor.extract_keywords(
                prompt=prompt,
                max_keywords=max_keywords,
                use_pos_filtering=True,
                use_ner_filtering=True,
                use_semantic_expansion=True,
                similarity_threshold=0.7,
                expansion_threshold=0.8
            )
            
            # Extract keywords from result
            keywords = [kw['keyword'] for kw in keyword_result['extracted_keywords']]
            
            # Filter and clean keywords
            filtered_keywords = self._filter_keywords(keywords)
            
            logger.info(f"Module 1 extracted {len(keywords)} keywords, filtered to {len(filtered_keywords)}")
            return filtered_keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords from Module 1: {e}")
            # Fallback to basic keyword extraction
            return self._basic_keyword_extraction(prompt)
    
    def _filter_keywords(self, keywords: List[str]) -> List[str]:
        """
        Filter and clean keywords to remove duplicates and low-quality terms
        """
        filtered = []
        seen = set()
        
        # Common stop words to remove
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 
            'may', 'might', 'can', 'this', 'that', 'these', 'those', 'research'
        }
        
        for keyword in keywords:
            # Clean the keyword
            clean_keyword = keyword.lower().strip()
            
            # Skip if too short or stop word
            if len(clean_keyword) < 3 or clean_keyword in stop_words:
                continue
                
            # Skip if already seen
            if clean_keyword in seen:
                continue
                
            # Skip if it's just a single common word repeated
            if clean_keyword.count(' ') == 0 and len(clean_keyword) < 5:
                continue
                
            # Skip if it's mostly the same word repeated
            words = clean_keyword.split()
            if len(set(words)) == 1 and len(words) > 1:
                continue
                
            # Skip if it contains repetitive patterns
            if self._has_repetitive_pattern(clean_keyword):
                continue
                
            filtered.append(keyword)
            seen.add(clean_keyword)
            
            # Limit to max 15 keywords
            if len(filtered) >= 15:
                break
        
        return filtered
    
    def _has_repetitive_pattern(self, keyword: str) -> bool:
        """
        Check if keyword has repetitive patterns
        """
        words = keyword.split()
        
        # Check for exact word repetition
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
            if word_counts[word] > 2:  # More than 2 repetitions
                return True
        
        # Check for similar word patterns
        if len(words) >= 3:
            for i in range(len(words) - 2):
                if words[i] == words[i+1] or words[i] == words[i+2]:
                    return True
        
        return False
    
    def _basic_keyword_extraction(self, prompt: str) -> List[str]:
        """
        Basic keyword extraction as fallback
        """
        # Simple word extraction
        words = prompt.lower().split()
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return list(set(keywords))[:10]  # Return unique keywords, max 10
    
    def _generate_search_queries(self, keywords: List[str]) -> List[str]:
        """
        Generate search queries from keywords using intelligent combinations
        """
        queries = []
        
        # Filter out very short or repetitive keywords
        good_keywords = [kw for kw in keywords if len(kw.split()) >= 1 and len(kw) > 3]
        
        # Single keyword queries (only meaningful ones)
        for keyword in good_keywords[:8]:
            if len(keyword.split()) >= 2:  # Prefer multi-word keywords
                queries.append(keyword)
        
        # Two-word combinations (more meaningful)
        for i in range(min(6, len(good_keywords))):
            for j in range(i + 1, min(i + 4, len(good_keywords))):
                if i != j:
                    combined = f"{good_keywords[i]} {good_keywords[j]}"
                    if len(combined) > 10:  # Only meaningful combinations
                        queries.append(combined)
        
        # Add original prompt keywords if they're good
        if len(queries) < 10:
            for keyword in good_keywords:
                if keyword not in queries and len(keyword) > 5:
                    queries.append(keyword)
        
        # Remove duplicates and limit
        unique_queries = list(set(queries))
        # Sort by length (prefer longer, more specific queries)
        unique_queries.sort(key=len, reverse=True)
        
        logger.info(f"Generated {len(unique_queries)} search queries from {len(keywords)} keywords")
        return unique_queries[:15]  # Limit to 15 queries
    
    async def search_sources(self, request: SearchRequest) -> SearchResponse:
        """
        Search for sources using real SERP API
        """
        # Try to initialize if not already initialized
        if not self.initialized:
            try:
                logger.info("🔄 Initializing Module 2 service...")
                await self.initialize()
                logger.info("✅ Module 2 service initialized successfully")
            except Exception as e:
                logger.error(f"❌ Service initialization failed: {e}")
                raise RuntimeError(f"Failed to initialize Dynamic Source Discovery service: {e}")
        
        # Validate SerpApi configuration
        logger.info(f"🔍 Checking SerpApi configuration...")
        logger.info(f"   SerpApi enabled: {self.serpapi_enabled}")
        logger.info(f"   SerpApi key: {self.serpapi_key[:10] if self.serpapi_key else 'None'}...")
        
        if not self.serpapi_enabled or not self.serpapi_key:
            logger.error("❌ SerpApi is not properly configured")
            raise RuntimeError("SerpApi is not properly configured. Please set SERPAPI_KEY in your .env file.")
        
        start_time = time.time()
        logger.info(f"🚀 Starting search with {len(request.keywords)} keywords")
        
        # Generate search queries
        logger.info(f"🔧 Generating search queries from {len(request.keywords)} keywords...")
        search_queries = self._generate_search_queries(request.keywords)
        logger.info(f"✅ Generated {len(search_queries)} search queries: {search_queries[:3]}...")
        
        all_results = []
        
        # Perform SERP API calls
        logger.info(f"🌐 Starting SERP API calls for {len(search_queries)} queries...")
        for i, query in enumerate(search_queries):
            try:
                logger.info(f"🔍 Processing query {i+1}/{len(search_queries)}: '{query}'")
                # Perform real search using SerpApi
                query_results = await self._perform_serp_search(
                    query, request.max_results // len(search_queries), request
                )
                all_results.extend(query_results)
                logger.info(f"✅ Query {i+1} completed: {len(query_results)} results")
                
                # Add delay to avoid rate limiting
                await asyncio.sleep(1.0)  # Increased delay for real API
                
            except RuntimeError as e:
                # SerpApi configuration or API errors
                logger.error(f"❌ SerpApi error for query '{query}': {e}")
                raise e  # Re-raise SerpApi errors
            except Exception as e:
                logger.error(f"❌ Unexpected error searching for query '{query}': {e}")
                continue
        
        # Remove duplicates and rank by relevance
        unique_results = self._deduplicate_results(all_results)
        ranked_results = self._rank_search_results(unique_results, request.keywords)
        
        # Limit results
        final_results = ranked_results[:request.max_results]
        
        processing_time = time.time() - start_time
        
        # Save to database (optional)
        if self.db_session:
            try:
                await self._save_search_results(final_results)
            except Exception as e:
                logger.warning(f"Failed to save search results to database: {e}")
        
        response = SearchResponse(
            search_request=request,
            total_results=len(final_results),
            search_results=final_results,
            processing_time=processing_time,
            search_stats={
                'queries_generated': len(search_queries),
                'unique_results': len(unique_results),
                'duplicates_removed': len(all_results) - len(unique_results)
            }
        )
        
        logger.info(f"Search completed: {len(final_results)} results in {processing_time:.2f}s")
        return response
    
    async def _perform_serp_search(self, query: str, max_results: int, request: SearchRequest) -> List[SearchResult]:
        """
        Perform real SERP API search using SerpApi
        """
        logger.info(f"=== SERP API SEARCH DEBUG ===")
        logger.info(f"Query: {query}")
        logger.info(f"Max results: {max_results}")
        logger.info(f"SerpApi enabled: {self.serpapi_enabled}")
        logger.info(f"SerpApi key: {self.serpapi_key[:10] if self.serpapi_key else 'None'}...")
        
        if not self.serpapi_enabled:
            raise RuntimeError("SerpApi is not enabled. Please configure SERPAPI_KEY in your .env file.")
        
        if not self.serpapi_key or self.serpapi_key == "your_serpapi_key_here":
            raise RuntimeError("SerpApi key is not properly configured. Please set a valid SERPAPI_KEY in your .env file.")
        
        logger.info(f"✅ Using real SerpApi with key: {self.serpapi_key[:10]}...")
        
        # Prepare search parameters
        search_params = {
            "q": query,
            "api_key": self.serpapi_key,
            "num": min(max_results, 100),  # SerpApi limit
            "hl": request.language,
            "gl": request.country,
            "safe": "active"
        }
        
        # Add time filter if specified
        if request.time_filter:
            if request.time_filter == TimeFilter.PAST_24_HOURS:
                search_params["tbs"] = "qdr:d"
            elif request.time_filter == TimeFilter.PAST_WEEK:
                search_params["tbs"] = "qdr:w"
            elif request.time_filter == TimeFilter.PAST_MONTH:
                search_params["tbs"] = "qdr:m"
            elif request.time_filter == TimeFilter.PAST_3_MONTHS:
                search_params["tbs"] = "qdr:3m"
            elif request.time_filter == TimeFilter.PAST_YEAR:
                search_params["tbs"] = "qdr:y"
            elif request.time_filter == TimeFilter.CUSTOM_RANGE:
                if request.custom_date_start and request.custom_date_end:
                    start_date = request.custom_date_start.strftime("%m/%d/%Y")
                    end_date = request.custom_date_end.strftime("%m/%d/%Y")
                    search_params["tbs"] = f"cdr:1,cd_min:{start_date},cd_max:{end_date}"
        
        try:
            # Perform search
            search = GoogleSearch(search_params)
            results = search.get_dict()
            
            search_results = []
            
            # Extract organic results
            if "organic_results" in results:
                for i, result in enumerate(results["organic_results"][:max_results]):
                    try:
                        url = result.get("link", "")
                        title = result.get("title", "")
                        snippet = result.get("snippet", "")
                        domain = urlparse(url).netloc if url else ""
                        
                        # Calculate relevance score
                        relevance_score = self._calculate_relevance_score(query, request.keywords)
                        
                        # Try to extract date
                        estimated_date = None
                        if "date" in result:
                            try:
                                estimated_date = datetime.fromisoformat(result["date"].replace("Z", "+00:00"))
                            except:
                                pass
                        
                        search_result = SearchResult(
                            url=url,
                            title=title,
                            snippet=snippet,
                            domain=domain,
                            relevance_score=relevance_score,
                            estimated_date=estimated_date,
                            search_query=query
                        )
                        search_results.append(search_result)
                        
                    except Exception as e:
                        logger.error(f"Error processing search result {i}: {e}")
                        continue
            
            logger.info(f"SerpApi search completed: {len(search_results)} results found")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in SerpApi search: {e}")
            raise RuntimeError(f"SerpApi search failed: {e}")
    
    def _calculate_relevance_score(self, query: str, keywords: List[str]) -> float:
        """Calculate relevance score based on keyword matches"""
        query_lower = query.lower()
        keyword_matches = sum(1 for kw in keywords if kw.lower() in query_lower)
        return min(keyword_matches / len(keywords), 1.0)
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on URL"""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        return unique_results
    
    def _rank_search_results(self, results: List[SearchResult], keywords: List[str]) -> List[SearchResult]:
        """Rank search results by relevance and domain reputation"""
        for result in results:
            # Get domain reputation
            domain_reputation = self._get_domain_reputation(result.domain)
            
            # Adjust relevance score based on domain reputation
            result.relevance_score = (result.relevance_score * 0.7 + domain_reputation * 0.3)
            result.relevance_score = max(0.0, min(1.0, result.relevance_score))
        
        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results
    
    def _get_domain_reputation(self, domain: str) -> float:
        """Get domain reputation score"""
        if domain in self.trusted_domains:
            return self.trusted_domains[domain]['score']
        return 0.5  # Default score for unknown domains
    
    async def _save_search_results(self, results: List[SearchResult]):
        """Save search results to database"""
        try:
            for result in results:
                db_result = SearchResultDB(
                    url=result.url,
                    title=result.title,
                    snippet=result.snippet,
                    domain=result.domain,
                    relevance_score=result.relevance_score,
                    estimated_date=result.estimated_date,
                    search_query=result.search_query
                )
                self.db_session.add(db_result)
            
            self.db_session.commit()
            logger.info(f"Saved {len(results)} search results to database")
            
        except Exception as e:
            logger.error(f"Error saving search results: {e}")
            self.db_session.rollback()
    
    async def crawl_sources(self, request: CrawlRequest) -> CrawlResponse:
        """
        Crawl discovered sources using intelligent link prioritization
        """
        if not self.initialized:
            raise RuntimeError("Service not initialized")
        
        start_time = time.time()
        logger.info(f"Starting crawl with {len(request.seed_urls)} seed URLs")
        
        crawled_pages = []
        failed_crawls = 0
        discovered_domains = set()
        
        # Initialize crawl queue
        crawl_queue = [(url, 0) for url in request.seed_urls]  # (url, depth)
        crawled_urls = set()
        
        while crawl_queue and len(crawled_pages) < request.max_pages:
            # Get next URL to crawl
            current_url, current_depth = crawl_queue.pop(0)
            
            if current_url in crawled_urls:
                continue
            
            try:
                # Crawl the page
                page_data = await self._crawl_page(
                    current_url, current_depth, request
                )
                
                if page_data:
                    crawled_pages.append(page_data)
                    crawled_urls.add(current_url)
                    discovered_domains.add(page_data.domain)
                    
                    # Add discovered links to queue if within depth limit
                    if current_depth < request.crawl_depth.value:
                        new_links = self._prioritize_links(
                            page_data.discovered_links, 
                            request.keywords, 
                            current_depth + 1
                        )
                        
                        for link_info in new_links:
                            if link_info.should_crawl and link_info.url not in crawled_urls:
                                crawl_queue.append((link_info.url, current_depth + 1))
                    
                    # Add delay between requests
                    await asyncio.sleep(request.delay_between_requests)
                else:
                    failed_crawls += 1
                    
            except Exception as e:
                logger.error(f"Error crawling {current_url}: {e}")
                failed_crawls += 1
                continue
        
        processing_time = time.time() - start_time
        
        # Save crawled pages to database
        await self._save_crawled_pages(crawled_pages)
        
        response = CrawlResponse(
            crawl_request=request,
            total_pages_crawled=len(crawled_pages),
            successful_crawls=len(crawled_pages),
            failed_crawls=failed_crawls,
            crawled_pages=crawled_pages,
            processing_time=processing_time,
            crawl_stats={
                'average_response_time': sum(p.response_time for p in crawled_pages) / len(crawled_pages) if crawled_pages else 0,
                'average_relevance_score': sum(p.relevance_score for p in crawled_pages) / len(crawled_pages) if crawled_pages else 0,
                'max_depth_reached': max(p.crawl_depth for p in crawled_pages) if crawled_pages else 0
            },
            discovered_domains=list(discovered_domains)
        )
        
        logger.info(f"Crawl completed: {len(crawled_pages)} pages in {processing_time:.2f}s")
        return response
    
    async def _crawl_page(self, url: str, depth: int, request: CrawlRequest) -> Optional[CrawledPage]:
        """Crawl a single page"""
        try:
            start_time = time.time()
            
            # Prepare headers
            headers = {
                'User-Agent': np.random.choice(self.user_agents) if request.user_agent_rotation else self.user_agents[0],
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # Make request
            response = requests.get(
                url, 
                headers=headers, 
                timeout=request.timeout,
                allow_redirects=True
            )
            
            response_time = time.time() - start_time
            
            if response.status_code != 200:
                logger.warning(f"HTTP {response.status_code} for {url}")
                return None
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No title"
            
            # Extract content (simplified)
            content = self._extract_text_content(soup)
            
            # Extract links
            discovered_links = self._extract_links(soup, url, request)
            
            # Calculate relevance score
            relevance_score = self._calculate_page_relevance(content, request.keywords)
            
            # Extract metadata
            meta_data = self._extract_metadata(soup)
            
            page_data = CrawledPage(
                url=url,
                title=title_text,
                content=content,
                html_content=response.text,
                domain=urlparse(url).netloc,
                crawl_depth=depth,
                relevance_score=relevance_score,
                discovered_links=discovered_links,
                meta_data=meta_data,
                crawl_timestamp=datetime.now(),
                response_time=response_time,
                status_code=response.status_code
            )
            
            return page_data
            
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            return None
    
    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract clean text content from HTML"""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:10000]  # Limit content length
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str, request: CrawlRequest) -> List[str]:
        """Extract and filter links from page"""
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Convert relative URLs to absolute
            if href.startswith('/'):
                href = urljoin(base_url, href)
            elif not href.startswith('http'):
                continue
            
            # Filter links
            if self._should_skip_link(href, request):
                continue
            
            links.append(href)
        
        return list(set(links))  # Remove duplicates
    
    def _should_skip_link(self, url: str, request: CrawlRequest) -> bool:
        """Check if link should be skipped"""
        parsed = urlparse(url)
        
        # Skip file extensions
        if any(parsed.path.lower().endswith(ext) for ext in self.skip_extensions):
            return True
        
        # Skip social media
        if parsed.netloc in self.social_media_domains:
            return True
        
        # Skip external links if not following them
        if not request.follow_external_links and parsed.netloc != urlparse(request.seed_urls[0]).netloc:
            return True
        
        return False
    
    def _calculate_page_relevance(self, content: str, keywords: List[str]) -> float:
        """Calculate page relevance based on keyword presence"""
        content_lower = content.lower()
        
        # Count keyword matches
        keyword_matches = sum(1 for kw in keywords if kw.lower() in content_lower)
        
        # Calculate score
        score = keyword_matches / len(keywords) if keywords else 0.0
        
        # Boost score for multiple occurrences
        total_occurrences = sum(content_lower.count(kw.lower()) for kw in keywords)
        if total_occurrences > 0:
            score += min(total_occurrences / 100, 0.3)  # Cap boost at 0.3
        
        return max(0.0, min(1.0, score))
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract metadata from page"""
        metadata = {}
        
        # Meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', meta.get('property', ''))
            content = meta.get('content', '')
            if name and content:
                metadata[name] = content
        
        # Open Graph tags
        og_tags = {}
        for meta in soup.find_all('meta', property=re.compile(r'^og:')):
            property_name = meta.get('property', '')
            content = meta.get('content', '')
            if property_name and content:
                og_tags[property_name] = content
        
        if og_tags:
            metadata['open_graph'] = og_tags
        
        return metadata
    
    def _prioritize_links(self, links: List[str], keywords: List[str], depth: int) -> List[LinkPrioritization]:
        """Prioritize links for crawling"""
        prioritized_links = []
        
        for link in links:
            # Calculate priority score
            keyword_matches = [kw for kw in keywords if kw.lower() in link.lower()]
            keyword_score = len(keyword_matches) / len(keywords) if keywords else 0.0
            
            # Domain reputation
            domain = urlparse(link).netloc
            domain_reputation = self._get_domain_reputation(domain)
            
            # Semantic similarity (simplified)
            semantic_similarity = 0.5  # Placeholder
            
            # Calculate final priority score
            priority_score = (
                keyword_score * 0.4 +
                domain_reputation * 0.3 +
                semantic_similarity * 0.2 +
                (1.0 / (depth + 1)) * 0.1  # Prefer shallower depth
            )
            
            # Determine if should crawl
            should_crawl = (
                priority_score > 0.3 and  # Minimum threshold
                depth < 4 and  # Maximum depth
                domain_reputation > 0.3  # Minimum domain reputation
            )
            
            link_info = LinkPrioritization(
                url=link,
                priority_score=priority_score,
                keyword_matches=keyword_matches,
                semantic_similarity=semantic_similarity,
                domain_reputation=domain_reputation,
                crawl_depth=depth,
                should_crawl=should_crawl
            )
            
            prioritized_links.append(link_info)
        
        # Sort by priority score
        prioritized_links.sort(key=lambda x: x.priority_score, reverse=True)
        return prioritized_links
    
    async def _save_crawled_pages(self, pages: List[CrawledPage]):
        """Save crawled pages to database"""
        try:
            for page in pages:
                db_page = CrawledPageDB(
                    url=page.url,
                    title=page.title,
                    content=page.content,
                    html_content=page.html_content,
                    domain=page.domain,
                    crawl_depth=page.crawl_depth,
                    relevance_score=page.relevance_score,
                    discovered_links=str(page.discovered_links),  # Convert list to string
                    meta_data=str(page.meta_data),  # Convert dict to string
                    crawl_timestamp=page.crawl_timestamp,
                    response_time=page.response_time,
                    status_code=page.status_code
                )
                self.db_session.add(db_page)
            
            self.db_session.commit()
            logger.info(f"Saved {len(pages)} crawled pages to database")
            
        except Exception as e:
            logger.error(f"Error saving crawled pages: {e}")
            self.db_session.rollback()
    
    async def get_discovery_stats(self) -> DiscoveryStats:
        """Get discovery statistics"""
        try:
            # Get counts from database
            total_searches = self.db_session.query(SearchResultDB).count()
            total_crawls = self.db_session.query(CrawledPageDB).count()
            
            # Get unique domains
            unique_domains = self.db_session.query(CrawledPageDB.domain).distinct().count()
            
            # Calculate average relevance score
            avg_relevance = self.db_session.query(CrawledPageDB.relevance_score).all()
            avg_relevance_score = sum(score[0] for score in avg_relevance) / len(avg_relevance) if avg_relevance else 0.0
            
            stats = DiscoveryStats(
                total_searches=total_searches,
                total_crawls=total_crawls,
                total_pages_discovered=total_crawls,
                unique_domains=unique_domains,
                average_relevance_score=avg_relevance_score,
                processing_time_total=0.0,  # Would need to track this
                success_rate=95.0  # Placeholder
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting discovery stats: {e}")
            raise


# Global instance
discovery_service = DynamicSourceDiscovery() 