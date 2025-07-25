from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
import logging

from .schemas import TextQualityRequest, TextQualityResponse
from .services import text_quality_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/text-quality", tags=["Text Quality Analysis"])


@router.post("/analyze-from-prompt", response_model=TextQualityResponse)
async def analyze_text_quality_from_prompt(request: TextQualityRequest):
    """
    Ana Endpoint: Prompt'tan başlayarak tüm modülleri zincirleme çalıştırır ve metin nitelikleri analizi ekler
    
    Bu endpoint şu adımları takip eder:
    1. Modül 1: Prompt'tan anahtar kelime çıkarma
    2. Modül 2: Anahtar kelimelerle URL keşfi
    3. Modül 3: URL'lerdeki içerikleri analiz etme
    4. Modül 4: İçeriklere metin nitelikleri analizi ekleme
    
    Returns:
        TextQualityResponse: Tüm modüllerin sonuçları ve geliştirilmiş içerikler
    """
    try:
        logger.info(f"📊 Metin nitelikleri analizi başlatılıyor: {request.prompt[:100]}...")
        
        request_dict = request.dict()
        
        result = await text_quality_service.analyze_text_quality_from_prompt(request_dict)
        
        logger.info(f"✅ Metin nitelikleri analizi tamamlandı. {result.relevant_articles} alakalı makale bulundu.")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Metin nitelikleri analizi hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Metin nitelikleri analizi hatası: {str(e)}")


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "module": "Text Quality Analysis",
        "version": "1.0.0",
        "description": "Gelişmiş metin nitelikleri analizi modülü"
    }


@router.post("/analyze-single-text")
async def analyze_single_text(request: Dict[str, str]):
    try:
        text = request.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Text field is required")
            
        logger.info(f"📝 Tek metin analizi başlatılıyor: {text[:100]}...")
        
        # Tek metin analizi
        analysis = await text_quality_service._analyze_text_quality(text)
        
        logger.info("✅ Tek metin analizi tamamlandı.")
        
        return analysis
        
    except Exception as e:
        logger.error(f"❌ Tek metin analizi hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Tek metin analizi hatası: {str(e)}")
    """
    Tek bir metin için metin nitelikleri analizi yapar
    
    Args:
        text: Analiz edilecek metin
        
    Returns:
        TextQualityAnalysis: Metin nitelikleri analizi sonucu
    """
    try:
        logger.info(f"📝 Tek metin analizi başlatılıyor: {text[:100]}...")
        
        # Tek metin analizi
        analysis = await text_quality_service._analyze_text_quality(text)
        
        logger.info("✅ Tek metin analizi tamamlandı.")
        
        return analysis
        
    except Exception as e:
        logger.error(f"❌ Tek metin analizi hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Tek metin analizi hatası: {str(e)}")


@router.get("/stats")
async def get_analysis_stats():
    return {
        "module": "Text Quality Analysis",
        "features": [
            "Sentiment Analysis (Hugging Face + NLTK VADER)",
            "Objectivity/Subjectivity Analysis",
            "Readability Scores (TextStat)",
            "Text Metrics (SpaCy)",
            "Full Pipeline Integration (Module 1 → 2 → 3 → 4)"
        ],
        "supported_languages": ["English"],
        "analysis_types": [
            "Sentiment (Positive/Negative/Neutral)",
            "Objectivity (Objective/Subjective/Mixed)",
            "Readability (Flesch, Gunning Fog, SMOG, etc.)",
            "Text Metrics (Word count, sentence length, etc.)"
        ]
    } 