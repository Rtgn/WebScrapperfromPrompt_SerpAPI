"""
FastAPI routes for Module 2: Dynamic Source Discovery
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import List, Optional
import logging

from .schemas import (
    SearchRequest, SearchResponse, CrawlRequest, CrawlResponse,
    DiscoveryStats, DomainReputation, LinkPrioritization,
    TimeFilter, SearchEngine
)
from .services import discovery_service

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/module2", tags=["Dynamic Source Discovery"])


@router.on_event("startup")
async def startup_event():
    try:
        await discovery_service.initialize()
        logger.info("Module 2 discovery service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Module 2 discovery service: {e}")
        raise


@router.post("/search", response_model=SearchResponse, summary="Search for sources")
async def search_sources(request: SearchRequest):
    """
    Search for sources using intelligent query generation and SERP API
    
    This endpoint takes keywords from Module 1 and generates intelligent search queries
    to find relevant sources across multiple search engines.
    
    **Features:**
    - Intelligent query generation from keywords
    - Multi-search engine support (Google, Bing, DuckDuckGo)
    - Time-based filtering
    - Domain reputation scoring
    - Relevance ranking
    """
    try:
        logger.info(f"Search request received with {len(request.keywords)} keywords")
        
        response = await discovery_service.search_sources(request)
        
        logger.info(f"Search completed: {response.total_results} results found")
        return response
        
    except Exception as e:
        logger.error(f"Error in search_sources: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/search-from-prompt", response_model=SearchResponse, summary="Search for sources from prompt")
async def search_from_prompt(
    prompt: str,
    max_keywords: int = 20,
    max_results: int = 50,
    time_filter: Optional[TimeFilter] = TimeFilter.PAST_MONTH,
    search_engine: SearchEngine = SearchEngine.GOOGLE,
    language: str = "en",
    country: str = "us"
):
    try:
        logger.info(f"Search from prompt request received: {prompt[:100]}...")
        
        # Extract keywords from prompt using Module 1
        keywords = await discovery_service._extract_keywords_from_prompt(prompt, max_keywords)
        
        if not keywords:
            raise HTTPException(status_code=400, detail="No keywords extracted from prompt")
        
        # Create search request
        search_request = SearchRequest(
            keywords=keywords,
            max_results=max_results,
            time_filter=time_filter,
            search_engine=search_engine,
            language=language,
            country=country
        )
        
        # Perform search
        response = await discovery_service.search_sources(search_request)
        
        # Add keyword extraction info to response
        response.search_stats['keywords_extracted'] = len(keywords)
        response.search_stats['original_prompt'] = prompt
        
        logger.info(f"Search from prompt completed: {response.total_results} results found")
        return response
        
    except Exception as e:
        logger.error(f"Error in search_from_prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Search from prompt failed: {str(e)}")


@router.post("/crawl", response_model=CrawlResponse, summary="Crawl discovered sources")
async def crawl_sources(request: CrawlRequest):
    """
    Crawl discovered sources using intelligent link prioritization
    
    This endpoint crawls web pages starting from seed URLs, using intelligent
    link prioritization and filtering to discover relevant content.
    
    **Features:**
    - Intelligent link prioritization
    - Bot detection avoidance
    - Proxy and User-Agent rotation
    - JavaScript rendering support
    - Depth-limited crawling
    - Relevance scoring
    """
    try:
        logger.info(f"Crawl request received with {len(request.seed_urls)} seed URLs")
        
        response = await discovery_service.crawl_sources(request)
        
        logger.info(f"Crawl completed: {response.total_pages_crawled} pages crawled")
        return response
        
    except Exception as e:
        logger.error(f"Error in crawl_sources: {e}")
        raise HTTPException(status_code=500, detail=f"Crawl failed: {str(e)}")


@router.post("/search-and-crawl", response_model=dict, summary="Search and crawl in one operation")
async def search_and_crawl(
    keywords: List[str],
    max_search_results: int = 50,
    max_crawl_pages: int = 100,
    crawl_depth: int = 2,
    background_tasks: BackgroundTasks = None
):
    """
    Combined search and crawl operation
    
    This endpoint performs both search and crawl operations in sequence,
    using the search results as seed URLs for crawling.
    
    **Process:**
    1. Generate search queries from keywords
    2. Search for relevant sources
    3. Use search results as seed URLs for crawling
    4. Return combined results
    """
    try:
        logger.info(f"Search and crawl request received with {len(keywords)} keywords")
        
        # Step 1: Search for sources
        search_request = SearchRequest(
            keywords=keywords,
            max_results=max_search_results
        )
        
        search_response = await discovery_service.search_sources(search_request)
        
        if not search_response.search_results:
            raise HTTPException(status_code=404, detail="No search results found")
        
        # Step 2: Extract URLs for crawling
        seed_urls = [result.url for result in search_response.search_results]
        
        # Step 3: Crawl discovered sources
        crawl_request = CrawlRequest(
            seed_urls=seed_urls,
            keywords=keywords,
            max_pages=max_crawl_pages,
            crawl_depth=crawl_depth
        )
        
        crawl_response = await discovery_service.crawl_sources(crawl_request)
        
        # Step 4: Combine results
        combined_response = {
            "search_results": search_response,
            "crawl_results": crawl_response,
            "total_processing_time": search_response.processing_time + crawl_response.processing_time,
            "summary": {
                "keywords_processed": len(keywords),
                "sources_discovered": search_response.total_results,
                "pages_crawled": crawl_response.total_pages_crawled,
                "unique_domains": len(crawl_response.discovered_domains),
                "average_relevance_score": sum(p.relevance_score for p in crawl_response.crawled_pages) / len(crawl_response.crawled_pages) if crawl_response.crawled_pages else 0
            }
        }
        
        logger.info(f"Search and crawl completed successfully")
        return combined_response
        
    except Exception as e:
        logger.error(f"Error in search_and_crawl: {e}")
        raise HTTPException(status_code=500, detail=f"Search and crawl failed: {str(e)}")


@router.get("/stats", response_model=DiscoveryStats, summary="Get discovery statistics")
async def get_discovery_stats():
    try:
        stats = await discovery_service.get_discovery_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting discovery stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/domain-reputation/{domain}", response_model=DomainReputation, summary="Get domain reputation")
async def get_domain_reputation(domain: str):
    try:
        reputation_score = discovery_service._get_domain_reputation(domain)
        
        # Determine if domain is trusted
        is_trusted = reputation_score > 0.7
        
        # Determine category
        if domain in discovery_service.trusted_domains:
            category = discovery_service.trusted_domains[domain]['category']
        else:
            category = "unknown"
        
        reputation = DomainReputation(
            domain=domain,
            reputation_score=reputation_score,
            is_trusted=is_trusted,
            category=category,
            last_checked=datetime.now()
        )
        
        return reputation
        
    except Exception as e:
        logger.error(f"Error getting domain reputation for {domain}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get domain reputation: {str(e)}")


@router.post("/prioritize-links", response_model=List[LinkPrioritization], summary="Prioritize links for crawling")
async def prioritize_links(
    links: List[str],
    keywords: List[str],
    depth: int = 1
):
    try:
        prioritized_links = discovery_service._prioritize_links(links, keywords, depth)
        return prioritized_links
        
    except Exception as e:
        logger.error(f"Error prioritizing links: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to prioritize links: {str(e)}")


@router.get("/health", summary="Health check")
async def health_check():
    try:
        health_status = {
            "status": "healthy",
            "module": "Dynamic Source Discovery",
            "initialized": discovery_service.initialized,
            "components": {
                "sentence_model": discovery_service.sentence_model is not None,
                "redis_client": discovery_service.redis_client is not None,
                "db_session": discovery_service.db_session is not None
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


# Import datetime for health check
from datetime import datetime 