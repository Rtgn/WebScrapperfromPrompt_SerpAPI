from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging
from typing import Dict, Any

from app.modules.module1.schemas import (
    KeywordExtractionRequest, 
    KeywordExtractionResponse, 
    KeywordExtractionError,
    KeywordInfo
)
from app.modules.module1.services import keyword_extractor
from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/module1", tags=["Module 1 - Advanced Keyword Extraction"])

@router.post("/extract-keywords", 
             response_model=KeywordExtractionResponse,
             summary="Extract keywords from text using advanced NLP techniques",
             description="""
             Extract keywords from input text using multiple advanced NLP techniques:
             
             **Technologies Used:**
             - **SpaCy**: Tokenization, lemmatization, POS tagging, and Named Entity Recognition (NER)
             - **KeyBERT**: BERT-based keyword extraction using sentence transformers
             - **Hugging Face Transformers**: Semantic understanding and embeddings
             - **Scikit-learn**: TF-IDF vectorization and cosine similarity calculations
             
             **Process Flow:**
             1. Text preprocessing with SpaCy
             2. Multi-method keyword extraction (SpaCy POS/NER, KeyBERT, TF-IDF)
             3. Keyword combination and ranking
             4. Semantic expansion (optional)
             5. Final ranking and deduplication
             6. Semantic clustering of related keywords
             
             **Features:**
             - Configurable extraction parameters
             - Multiple extraction sources combined
             - Semantic similarity clustering
             - Detailed extraction statistics
             - Processing time tracking
             """)
async def extract_keywords(request: KeywordExtractionRequest) -> KeywordExtractionResponse:
    try:
        logger.info(f"Processing keyword extraction request for prompt: {request.prompt[:100]}...")
        
        # Extract keywords using the advanced extractor
        result = keyword_extractor.extract_keywords(
            prompt=request.prompt,
            max_keywords=request.max_keywords,
            use_pos_filtering=request.use_pos_filtering,
            use_ner_filtering=request.use_ner_filtering,
            use_semantic_expansion=request.use_semantic_expansion,
            similarity_threshold=request.similarity_threshold,
            expansion_threshold=request.expansion_threshold
        )
        
        # Convert to response format
        keywords_info = []
        for kw in result['extracted_keywords']:
            keywords_info.append(KeywordInfo(
                keyword=kw['keyword'],
                score=kw['score'],
                source=kw['source'],
                pos_tag=kw.get('pos_tag'),
                entity_type=kw.get('entity_type'),
                frequency=kw.get('frequency')
            ))
        
        response = KeywordExtractionResponse(
            original_prompt=result['original_prompt'],
            extracted_keywords=keywords_info,
            total_keywords=result['total_keywords'],
            processing_time=result['processing_time'],
            extraction_stats=result['extraction_stats'],
            semantic_clusters=result.get('semantic_clusters')
        )
        
        logger.info(f"Successfully extracted {result['total_keywords']} keywords in {result['processing_time']:.2f} seconds")
        return response
        
    except Exception as e:
        logger.error(f"Error in keyword extraction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Keyword extraction failed: {str(e)}"
        )

@router.get("/health", 
            summary="Health check for Module 1",
            description="Check if Module 1 keyword extraction service is healthy and models are loaded")
async def health_check() -> Dict[str, Any]:

    try:
        # Check if the keyword extractor is initialized
        if not keyword_extractor.initialized:
            return {
                "status": "unhealthy",
                "module": "Module 1 - Advanced Keyword Extraction",
                "message": "Keyword extractor not initialized",
                "models_loaded": False
            }
        
        # Test basic functionality
        test_prompt = "This is a test prompt for health check."
        test_result = keyword_extractor.extract_keywords(
            prompt=test_prompt,
            max_keywords=5,
            use_semantic_expansion=False
        )
        
        return {
            "status": "healthy",
            "module": "Module 1 - Advanced Keyword Extraction",
            "message": "All models loaded and functioning correctly",
            "models_loaded": True,
            "test_extraction": {
                "keywords_extracted": test_result['total_keywords'],
                "processing_time": test_result['processing_time']
            },
            "configuration": {
                "spacy_model": settings.spacy_model,
                "keybert_model": settings.keybert_model,
                "max_keywords": settings.max_keywords,
                "min_keyword_length": settings.min_keyword_length
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "module": "Module 1 - Advanced Keyword Extraction",
            "message": f"Health check failed: {str(e)}",
            "models_loaded": False
        }

@router.get("/info", 
            summary="Get Module 1 information",
            description="Get detailed information about Module 1 capabilities and configuration")
async def get_module_info() -> Dict[str, Any]:

    return {
        "module_name": "Module 1 - Advanced Keyword Extraction",
        "version": settings.app_version,
        "description": """
        Advanced keyword extraction using multiple NLP techniques:
        - SpaCy for tokenization, lemmatization, POS tagging, and NER
        - KeyBERT for BERT-based keyword extraction
        - Hugging Face Transformers for semantic understanding
        - Scikit-learn for TF-IDF and cosine similarity
        """,
        "technologies": {
            "spacy": {
                "version": "3.7.2",
                "model": settings.spacy_model,
                "features": ["Tokenization", "Lemmatization", "POS Tagging", "NER"]
            },
            "keybert": {
                "version": "0.7.0",
                "model": settings.keybert_model,
                "features": ["BERT-based keyword extraction", "Sentence embeddings"]
            },
            "transformers": {
                "version": "4.36.0",
                "features": ["Semantic understanding", "Text embeddings"]
            },
            "scikit-learn": {
                "version": "1.3.2",
                "features": ["TF-IDF vectorization", "Cosine similarity"]
            }
        },
        "configuration": {
            "max_keywords": settings.max_keywords,
            "min_keyword_length": settings.min_keyword_length,
            "similarity_threshold": settings.similarity_threshold,
            "allowed_pos_tags": settings.allowed_pos_tags,
            "ner_entity_types": settings.ner_entity_types
        },
        "endpoints": {
            "extract_keywords": "POST /module1/extract-keywords",
            "health_check": "GET /module1/health",
            "info": "GET /module1/info"
        }
    } 