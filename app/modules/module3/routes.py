import logging
import uuid
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from .schemas import ContentAnalysisRequest, ContentAnalysisResponse
from .services import ContentAnalysisService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/content-analysis", tags=["Content Analysis"])


def get_content_analysis_service() -> ContentAnalysisService:
    return ContentAnalysisService()


@router.post("/analyze-from-prompt", response_model=ContentAnalysisResponse, summary="Prompt'tan içerik analizi")
async def analyze_content_from_prompt(
    request: ContentAnalysisRequest,
    service: ContentAnalysisService = Depends(get_content_analysis_service)
):
    try:
        logger.info(f"İçerik analizi isteği alındı: {request.prompt[:100]}...")
        
        # Request'i dictionary'e çevir
        request_dict = request.dict()
        
        # Ana analiz fonksiyonunu çağır
        results = await service.analyze_content_from_prompt(request_dict)
        
        logger.info(f"İçerik analizi tamamlandı. {results['relevant_articles']} alakalı makale bulundu.")
        
        return ContentAnalysisResponse(**results)
        
    except Exception as e:
        logger.error(f"İçerik analizi hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=f"İçerik analizi başarısız: {str(e)}")


@router.get("/health", summary="Sağlık kontrolü")
async def health_check():
    """Modül 3 sağlık kontrolü"""
    return {
        "status": "healthy",
        "module": "Modül 3: Akıllı İçerik Analizi ve Alaka Düzeyi Kontrolü",
        "version": "1.0.0",
        "description": "README algoritma akışını uygulayan tek endpoint modülü"
    } 