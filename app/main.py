from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import uvicorn
from contextlib import asynccontextmanager

from app.core.config import settings
from app.modules.module1.routes import router as module1_router
from app.modules.module2.routes import router as module2_router
from app.modules.module3.routes import router as module3_router
from app.modules.module4.routes import router as module4_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Advanced Keyword Extraction API...")
    logger.info(f"App Name: {settings.app_name}")
    logger.info(f"Version: {settings.app_version}")
    logger.info(f"Debug Mode: {settings.debug}")
    
    yield
    
    logger.info("Shutting down Advanced Keyword Extraction API...")

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    # Advanced Research API
    
    A modular FastAPI-based API for advanced research and content discovery using multiple AI techniques.
    
    ## Module 1: Advanced Keyword Extraction
    
    This module provides sophisticated keyword extraction capabilities using:
    
    - **SpaCy**: Tokenization, lemmatization, POS tagging, and Named Entity Recognition (NER)
    - **KeyBERT**: BERT-based keyword extraction using sentence transformers
    - **Hugging Face Transformers**: Semantic understanding and embeddings
    - **Scikit-learn**: TF-IDF vectorization and cosine similarity calculations
    
    ## Module 2: Dynamic Source Discovery
    
    This module provides intelligent web crawling and source discovery capabilities:
    
    - **SERP API**: Search engine result page data extraction
    - **Scrapy**: Advanced web crawling with bot detection avoidance
    - **Intelligent Link Prioritization**: Smart filtering and ranking of discovered links
    - **Domain Reputation**: Trust scoring for discovered sources
    - **Redis**: Queue management for scalable crawling
    - **PostgreSQL**: Data storage for discovered content
    
    ## Features
    
    - Multi-method keyword extraction
    - Semantic expansion and clustering
    - Intelligent source discovery
    - Bot detection avoidance
    - Configurable extraction parameters
    - Detailed extraction statistics
    - Processing time tracking
    
    ## Quick Start
    
    ### Module 1: Keyword Extraction
    1. **Health Check**: `GET /module1/health`
    2. **Module Info**: `GET /module1/info`
    3. **Extract Keywords**: `POST /module1/extract-keywords`
    
    ### Module 2: Source Discovery
    1. **Health Check**: `GET /module2/health`
    2. **Search Sources**: `POST /module2/search`
    3. **Crawl Sources**: `POST /module2/crawl`
    4. **Search and Crawl**: `POST /module2/search-and-crawl`
    
    ## Example Requests
    
    ### Keyword Extraction
    ```json
    {
        "prompt": "Yapay zeka etiği ve şeffaflık konularındaki son gelişmeler",
        "max_keywords": 20,
        "use_pos_filtering": true,
        "use_ner_filtering": true,
        "use_semantic_expansion": true,
        "similarity_threshold": 0.7,
        "expansion_threshold": 0.8
    }
    ```
    
    ### Source Discovery
    ```json
    {
        "keywords": ["artificial intelligence", "ethics", "transparency"],
        "max_results": 50,
        "time_filter": "past_month"
    }
    ```
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include module routers
app.include_router(module1_router)
app.include_router(module2_router)
app.include_router(module3_router)
app.include_router(module4_router)


@app.get("/", summary="API Root", description="Root endpoint with API information")
async def root():
    return {
        "app_name": settings.app_name,
        "version": settings.app_version,
        "description": "Advanced Keyword Extraction API using multiple NLP techniques",
        "modules": {
            "module1": {
                "name": "Advanced Keyword Extraction",
                "status": "active",
                "endpoints": [
                    "GET /module1/health",
                    "GET /module1/info", 
                    "POST /module1/extract-keywords"
                ]
            },
            "module2": {
                "name": "Dynamic Source Discovery",
                "status": "active",
                "endpoints": [
                    "GET /module2/health",
                    "POST /module2/search",
                    "POST /module2/crawl",
                    "POST /module2/search-and-crawl"
                ]
            },
            "module3": {
                "name": "Content Analysis and Relevance Control",
                "status": "active",
                "endpoints": [
                    "POST /api/v1/content-analysis/analyze-from-prompt",
                    "GET /api/v1/content-analysis/health"
                ]
            },
            "module4": {
                "name": "Advanced Text Quality Analysis",
                "status": "active",
                "endpoints": [
                    "POST /api/v1/text-quality/analyze-from-prompt",
                    "GET /api/v1/text-quality/health",
                    "POST /api/v1/text-quality/analyze-single-text",
                    "GET /api/v1/text-quality/stats"
                ]
            },

        },
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc"
        }
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "details": str(exc) if settings.debug else "Check server logs for details"
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info"
    ) 