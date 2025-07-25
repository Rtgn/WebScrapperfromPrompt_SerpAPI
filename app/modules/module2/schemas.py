"""
Pydantic schemas for Module 2: Dynamic Source Discovery
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
from enum import Enum


class SearchEngine(str, Enum):
    """Supported search engines"""
    GOOGLE = "google"
    BING = "bing"
    DUCKDUCKGO = "duckduckgo"


class TimeFilter(str, Enum):
    """Time-based filtering options"""
    PAST_24_HOURS = "past_24_hours"
    PAST_WEEK = "past_week"
    PAST_MONTH = "past_month"
    PAST_3_MONTHS = "past_3_months"
    PAST_YEAR = "past_year"
    CUSTOM_RANGE = "custom_range"


class CrawlDepth(int, Enum):
    """Crawl depth levels"""
    SHALLOW = 1
    MEDIUM = 2
    DEEP = 3
    VERY_DEEP = 4


class SearchRequest(BaseModel):
    """Request model for search operations"""
    keywords: List[str] = Field(..., description="List of keywords from Module 1")
    search_engine: SearchEngine = Field(default=SearchEngine.GOOGLE, description="Search engine to use")
    max_results: int = Field(default=50, ge=10, le=200, description="Maximum number of search results")
    time_filter: Optional[TimeFilter] = Field(default=TimeFilter.PAST_MONTH, description="Time-based filtering")
    custom_date_start: Optional[datetime] = Field(default=None, description="Custom start date for time filtering")
    custom_date_end: Optional[datetime] = Field(default=None, description="Custom end date for time filtering")
    language: str = Field(default="en", description="Search language")
    country: str = Field(default="us", description="Search country")
    
    @validator('custom_date_start', 'custom_date_end')
    def validate_custom_dates(cls, v, values):
        if 'time_filter' in values and values['time_filter'] == TimeFilter.CUSTOM_RANGE:
            if not v:
                raise ValueError("Custom dates required when using custom_range time filter")
        return v


class SearchResult(BaseModel):
    """Individual search result"""
    url: str = Field(..., description="Result URL")
    title: str = Field(..., description="Page title")
    snippet: str = Field(..., description="Page snippet/description")
    domain: str = Field(..., description="Domain name")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    estimated_date: Optional[datetime] = Field(default=None, description="Estimated publication date")
    search_query: str = Field(..., description="Search query that found this result")


class SearchResponse(BaseModel):
    """Response model for search operations"""
    search_request: SearchRequest = Field(..., description="Original search request")
    total_results: int = Field(..., description="Total number of results found")
    search_results: List[SearchResult] = Field(..., description="List of search results")
    processing_time: float = Field(..., description="Processing time in seconds")
    search_stats: Dict[str, Any] = Field(..., description="Search statistics")


class CrawlRequest(BaseModel):
    """Request model for crawling operations"""
    seed_urls: List[str] = Field(..., description="Initial URLs to crawl")
    keywords: List[str] = Field(..., description="Keywords for relevance filtering")
    max_pages: int = Field(default=100, ge=10, le=1000, description="Maximum pages to crawl")
    crawl_depth: CrawlDepth = Field(default=CrawlDepth.MEDIUM, description="Crawl depth level")
    follow_external_links: bool = Field(default=True, description="Whether to follow external links")
    use_proxies: bool = Field(default=True, description="Whether to use proxy rotation")
    respect_robots_txt: bool = Field(default=True, description="Whether to respect robots.txt")
    delay_between_requests: float = Field(default=1.0, ge=0.1, le=10.0, description="Delay between requests in seconds")
    timeout: int = Field(default=30, ge=10, le=120, description="Request timeout in seconds")
    user_agent_rotation: bool = Field(default=True, description="Whether to rotate user agents")
    javascript_rendering: bool = Field(default=True, description="Whether to render JavaScript")


class CrawledPage(BaseModel):
    """Individual crawled page"""
    url: str = Field(..., description="Page URL")
    title: str = Field(..., description="Page title")
    content: str = Field(..., description="Page content (cleaned text)")
    html_content: str = Field(..., description="Raw HTML content")
    domain: str = Field(..., description="Domain name")
    crawl_depth: int = Field(..., description="Crawl depth level")
    relevance_score: float = Field(..., description="Relevance score based on keywords")
    discovered_links: List[str] = Field(..., description="Links discovered on this page")
    meta_data: Dict[str, Any] = Field(..., description="Page metadata")
    crawl_timestamp: datetime = Field(..., description="When the page was crawled")
    response_time: float = Field(..., description="Response time in seconds")
    status_code: int = Field(..., description="HTTP status code")


class CrawlResponse(BaseModel):
    """Response model for crawling operations"""
    crawl_request: CrawlRequest = Field(..., description="Original crawl request")
    total_pages_crawled: int = Field(..., description="Total pages crawled")
    successful_crawls: int = Field(..., description="Number of successful crawls")
    failed_crawls: int = Field(..., description="Number of failed crawls")
    crawled_pages: List[CrawledPage] = Field(..., description="List of crawled pages")
    processing_time: float = Field(..., description="Total processing time in seconds")
    crawl_stats: Dict[str, Any] = Field(..., description="Crawl statistics")
    discovered_domains: List[str] = Field(..., description="List of discovered domains")


class DomainReputation(BaseModel):
    """Domain reputation information"""
    domain: str = Field(..., description="Domain name")
    reputation_score: float = Field(..., ge=0.0, le=1.0, description="Reputation score")
    is_trusted: bool = Field(..., description="Whether domain is trusted")
    category: str = Field(..., description="Domain category")
    last_checked: datetime = Field(..., description="When reputation was last checked")


class LinkPrioritization(BaseModel):
    """Link prioritization information"""
    url: str = Field(..., description="Link URL")
    priority_score: float = Field(..., ge=0.0, le=1.0, description="Priority score")
    keyword_matches: List[str] = Field(..., description="Matching keywords")
    semantic_similarity: float = Field(..., description="Semantic similarity score")
    domain_reputation: float = Field(..., description="Domain reputation score")
    crawl_depth: int = Field(..., description="Current crawl depth")
    should_crawl: bool = Field(..., description="Whether link should be crawled")


class DiscoveryStats(BaseModel):
    """Discovery statistics"""
    total_searches: int = Field(..., description="Total searches performed")
    total_crawls: int = Field(..., description="Total crawls performed")
    total_pages_discovered: int = Field(..., description="Total pages discovered")
    unique_domains: int = Field(..., description="Unique domains discovered")
    average_relevance_score: float = Field(..., description="Average relevance score")
    processing_time_total: float = Field(..., description="Total processing time")
    success_rate: float = Field(..., description="Success rate percentage") 