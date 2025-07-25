"""
Module 2: Dynamic Source Discovery and Extended Search
=====================================================

This module implements intelligent web crawling and source discovery using:
- SERP API for initial search results
- Scrapy for web crawling
- Dynamic content loading with Scrapy-Splash
- Proxy and User-Agent rotation
- Smart link prioritization and filtering
- Redis for queue management
- PostgreSQL for data storage

Key Features:
- Keyword-based search query generation
- Intelligent link discovery and prioritization
- Bot detection avoidance
- Scalable crawling architecture
- Time-based filtering
- Domain reputation checking
"""

from .routes import router
from .services import DynamicSourceDiscovery
from .schemas import SearchRequest, SearchResponse, CrawlRequest, CrawlResponse

__all__ = [
    'router',
    'DynamicSourceDiscovery', 
    'SearchRequest',
    'SearchResponse',
    'CrawlRequest',
    'CrawlResponse'
] 