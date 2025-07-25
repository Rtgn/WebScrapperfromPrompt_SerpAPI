from pydantic_settings import BaseSettings
from typing import Optional, List
import os

class Settings(BaseSettings):
    # Application settings
    app_name: str = "Advanced Research API"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # Module 1: Keyword Extraction settings
    spacy_model: str = "en_core_web_sm"
    keybert_model: str = "all-MiniLM-L6-v2"
    max_keywords: int = 20
    min_keyword_length: int = 3
    similarity_threshold: float = 0.7
    
    # Keyword extraction parameters
    use_pos_filtering: bool = True
    use_ner_filtering: bool = True
    use_semantic_expansion: bool = True
    expansion_threshold: float = 0.8
    
    # Allowed POS tags for keywords
    allowed_pos_tags: List[str] = [
        "NOUN", "PROPN", "ADJ", "VERB"
    ]
    
    # NER entity types to include
    ner_entity_types: List[str] = [
        "PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART"
    ]
    
    # Module 2: Dynamic Source Discovery settings
    serpapi_key: Optional[str] = None
    database_url: str = "sqlite:///discovery.db"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Crawling settings
    default_crawl_depth: int = 2
    default_max_pages: int = 100
    request_timeout: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Debug: Print SerpApi key status
if settings.serpapi_key:
    print(f"✅ SerpApi key loaded: {settings.serpapi_key[:10]}...")
else:
    print("❌ SerpApi key not found. Module 2 requires SERPAPI_KEY in .env file.")
    print("   Add SERPAPI_KEY=your_key to .env file for real API access.") 