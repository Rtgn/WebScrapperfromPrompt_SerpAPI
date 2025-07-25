"""
Modül 3: Akıllı İçerik Analizi ve Alaka Düzeyi Kontrolü - Şemalar
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime
from enum import Enum


class TimeFilter(str, Enum):
    """Zaman filtresi seçenekleri"""
    PAST_DAY = "past_day"
    PAST_WEEK = "past_week"
    PAST_MONTH = "past_month"
    PAST_YEAR = "past_year"
    ALL_TIME = "all_time"


class ContentAnalysisRequest(BaseModel):
    """İçerik analizi isteği şeması"""
    prompt: str = Field(..., description="Orijinal kullanıcı prompt'u")
    max_keywords: Optional[int] = Field(20, description="Maksimum anahtar kelime sayısı")
    max_urls: Optional[int] = Field(50, description="Maksimum URL sayısı")
    time_limit_days: Optional[int] = Field(30, description="Zaman sınırı (gün)")
    relevance_threshold: Optional[float] = Field(0.5, description="Alaka düzeyi eşiği (0-1 arası)")
    max_content_length: Optional[int] = Field(2048, description="Maksimum içerik uzunluğu (token)")
    excluded_domains: Optional[List[str]] = Field([], description="Hariç tutulacak domain'ler")
    required_keywords: Optional[List[str]] = Field([], description="Zorunlu anahtar kelimeler")
    language: Optional[str] = Field("en", description="Hedef dil")


class ExtractedContent(BaseModel):
    """Çıkarılan içerik şeması"""
    url: HttpUrl = Field(..., description="Kaynak URL")
    title: str = Field(..., description="Makale başlığı")
    content: str = Field(..., description="Temizlenmiş ana metin")
    summary: Optional[str] = Field(None, description="Özet veya giriş paragrafı")
    author: Optional[str] = Field(None, description="Yazar bilgisi")
    publish_date: Optional[datetime] = Field(None, description="Yayın tarihi")
    language: Optional[str] = Field(None, description="Tespit edilen dil")
    word_count: int = Field(..., description="Kelime sayısı")
    extraction_method: str = Field(..., description="Kullanılan çıkarma yöntemi")
    processing_time: float = Field(..., description="İşlem süresi (saniye)")


class ContentAnalysisResult(BaseModel):
    """İçerik analizi sonucu şeması"""
    content: ExtractedContent = Field(..., description="Çıkarılan içerik")
    relevance_score: float = Field(..., description="Alaka düzeyi skoru (0-1 arası)")
    is_relevant: bool = Field(..., description="Alaka düzeyi eşiğini geçiyor mu")
    is_within_time_limit: bool = Field(..., description="Zaman sınırı içinde mi")
    analysis_time: float = Field(..., description="Analiz süresi (saniye)")


class KeywordExtractionResult(BaseModel):
    """Anahtar kelime çıkarma sonucu şeması (Modül 1 taklidi)"""
    prompt: str = Field(..., description="Orijinal prompt")
    extracted_keywords: List[Dict[str, Any]] = Field(..., description="Çıkarılan anahtar kelimeler")
    total_keywords: int = Field(..., description="Toplam anahtar kelime sayısı")
    extraction_time: float = Field(..., description="Çıkarma süresi (saniye)")


class URLGenerationResult(BaseModel):
    """URL oluşturma sonucu şeması (Modül 2 taklidi)"""
    keywords: List[str] = Field(..., description="Kullanılan anahtar kelimeler")
    generated_urls: List[HttpUrl] = Field(..., description="Oluşturulan URL'ler")
    total_urls: int = Field(..., description="Toplam URL sayısı")
    generation_time: float = Field(..., description="Oluşturma süresi (saniye)")


class ContentAnalysisResponse(BaseModel):
    """İçerik analizi yanıt şeması"""
    # Modül 1 taklidi sonuçları
    module1_results: KeywordExtractionResult = Field(..., description="Anahtar kelime çıkarma sonuçları")
    
    # Modül 2 taklidi sonuçları
    module2_results: URLGenerationResult = Field(..., description="URL oluşturma sonuçları")
    
    # Modül 3 sonuçları
    module3_results: Dict[str, Any] = Field(..., description="İçerik analizi sonuçları")
    
    # Genel istatistikler
    total_urls_processed: int = Field(..., description="İşlenen toplam URL sayısı")
    relevant_articles: int = Field(..., description="Alakalı makale sayısı")
    average_relevance_score: float = Field(..., description="Ortalama alaka skoru")
    total_processing_time: float = Field(..., description="Toplam işlem süresi (saniye)")
    
    # Alakalı makaleler listesi
    relevant_content: List[ContentAnalysisResult] = Field(..., description="Alakalı içerikler")
    
    # Hata bilgileri
    errors: List[str] = Field([], description="İşlem sırasında oluşan hatalar") 