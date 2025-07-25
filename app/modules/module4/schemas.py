"""
Modül 4: Gelişmiş Metin Nitelikleri Analizi - Pydantic Şemaları
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class SentimentType(str, Enum):
    """Duygu türleri"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ObjectivityType(str, Enum):
    """Tarafsızlık türleri"""
    OBJECTIVE = "objective"
    SUBJECTIVE = "subjective"
    MIXED = "mixed"


class TextQualityRequest(BaseModel):
    """Metin nitelikleri analizi isteği"""
    prompt: str = Field(..., description="Araştırma prompt'u")
    max_keywords: int = Field(default=15, description="Maksimum anahtar kelime sayısı")
    max_urls: int = Field(default=20, description="Maksimum URL sayısı")
    time_limit_days: int = Field(default=30, description="Zaman sınırı (gün)")
    relevance_threshold: float = Field(default=0.1, description="Alaka düzeyi eşiği")
    max_content_length: int = Field(default=2048, description="Maksimum içerik uzunluğu")
    excluded_domains: List[str] = Field(default=[], description="Hariç tutulacak domainler")
    required_keywords: List[str] = Field(default=[], description="Zorunlu anahtar kelimeler")
    language: str = Field(default="en", description="Hedef dil")


class SentimentAnalysis(BaseModel):
    """Duygu analizi sonucu"""
    sentiment: SentimentType = Field(..., description="Duygu türü")
    confidence: float = Field(..., description="Güven skoru (0-1)")
    positive_score: float = Field(..., description="Pozitif skor")
    negative_score: float = Field(..., description="Negatif skor")
    neutral_score: float = Field(..., description="Nötr skor")


class ObjectivityAnalysis(BaseModel):
    """Tarafsızlık analizi sonucu"""
    objectivity_type: ObjectivityType = Field(..., description="Tarafsızlık türü")
    objectivity_score: float = Field(..., description="Tarafsızlık skoru (0-1)")
    subjectivity_score: float = Field(..., description="Subjektiflik skoru (0-1)")
    confidence: float = Field(..., description="Güven skoru")


class ReadabilityScores(BaseModel):
    """Okunabilirlik skorları"""
    flesch_reading_ease: float = Field(..., description="Flesch okuma kolaylığı")
    flesch_kincaid_grade: float = Field(..., description="Flesch-Kincaid seviye")
    gunning_fog: float = Field(..., description="Gunning Fog indeksi")
    smog_index: float = Field(..., description="SMOG indeksi")
    coleman_liau_index: float = Field(..., description="Coleman-Liau indeksi")
    automated_readability_index: float = Field(..., description="Otomatik okunabilirlik indeksi")
    dale_chall_score: float = Field(..., description="Dale-Chall skoru")
    linsear_write_formula: float = Field(..., description="Linsear Write formülü")
    fernandez_huerta: float = Field(..., description="Fernandez Huerta skoru")
    szigriszt_pazos: float = Field(..., description="Szigriszt Pazos indeksi")
    gutierrez_polini: float = Field(..., description="Gutierrez Polini indeksi")
    crawford: float = Field(..., description="Crawford formülü")


class TextMetrics(BaseModel):
    """Temel metin metrikleri"""
    word_count: int = Field(..., description="Kelime sayısı")
    sentence_count: int = Field(..., description="Cümle sayısı")
    character_count: int = Field(..., description="Karakter sayısı")
    paragraph_count: int = Field(..., description="Paragraf sayısı")
    average_word_length: float = Field(..., description="Ortalama kelime uzunluğu")
    average_sentence_length: float = Field(..., description="Ortalama cümle uzunluğu")
    unique_words: int = Field(..., description="Benzersiz kelime sayısı")
    vocabulary_diversity: float = Field(..., description="Kelime çeşitliliği (Type-Token Ratio)")


class TextQualityAnalysis(BaseModel):
    """Metin nitelikleri analizi sonucu"""
    sentiment_analysis: SentimentAnalysis = Field(..., description="Duygu analizi")
    objectivity_analysis: ObjectivityAnalysis = Field(..., description="Tarafsızlık analizi")
    readability_scores: ReadabilityScores = Field(..., description="Okunabilirlik skorları")
    text_metrics: TextMetrics = Field(..., description="Temel metin metrikleri")
    analysis_time: float = Field(..., description="Analiz süresi (saniye)")


class EnhancedContent(BaseModel):
    """Geliştirilmiş içerik (Modül 3'ten + Modül 4 analizi)"""
    url: str = Field(..., description="Makale URL'si")
    title: str = Field(..., description="Makale başlığı")
    content: str = Field(..., description="Makale içeriği")
    summary: str = Field(..., description="Makale özeti")
    author: Optional[str] = Field(None, description="Yazar")
    publish_date: Optional[datetime] = Field(None, description="Yayın tarihi")
    language: str = Field(..., description="Dil")
    word_count: int = Field(..., description="Kelime sayısı")
    extraction_method: str = Field(..., description="Çıkarma yöntemi")
    processing_time: float = Field(..., description="İşlem süresi")
    relevance_score: float = Field(..., description="Alaka düzeyi skoru")
    is_relevant: bool = Field(..., description="Alakalı mı?")
    is_within_time_limit: bool = Field(..., description="Zaman sınırı içinde mi?")
    text_quality_analysis: TextQualityAnalysis = Field(..., description="Metin nitelikleri analizi")


class Module1Result(BaseModel):
    """Modül 1 sonucu"""
    prompt: str = Field(..., description="Orijinal prompt")
    extracted_keywords: List[Dict[str, Any]] = Field(..., description="Çıkarılan anahtar kelimeler")
    total_keywords: int = Field(..., description="Toplam anahtar kelime sayısı")
    extraction_time: float = Field(..., description="Çıkarma süresi")


class Module2Result(BaseModel):
    """Modül 2 sonucu"""
    keywords: List[str] = Field(..., description="Kullanılan anahtar kelimeler")
    generated_urls: List[str] = Field(..., description="Oluşturulan URL'ler")
    total_urls: int = Field(..., description="Toplam URL sayısı")
    generation_time: float = Field(..., description="Oluşturma süresi")


class Module3Result(BaseModel):
    """Modül 3 sonucu"""
    total_urls_processed: int = Field(..., description="İşlenen toplam URL sayısı")
    relevant_articles: int = Field(..., description="Alakalı makale sayısı")
    average_relevance_score: float = Field(..., description="Ortalama alaka skoru")
    processing_time: float = Field(..., description="İşlem süresi")
    errors: List[str] = Field(default=[], description="Hatalar")


class TextQualityResponse(BaseModel):
    """Metin nitelikleri analizi yanıtı"""
    module1_results: Module1Result = Field(..., description="Modül 1 sonuçları")
    module2_results: Module2Result = Field(..., description="Modül 2 sonuçları")
    module3_results: Module3Result = Field(..., description="Modül 3 sonuçları")
    module4_results: Dict[str, Any] = Field(..., description="Modül 4 sonuçları")
    total_urls_processed: int = Field(..., description="İşlenen toplam URL sayısı")
    relevant_articles: int = Field(..., description="Alakalı makale sayısı")
    average_relevance_score: float = Field(..., description="Ortalama alaka skoru")
    total_processing_time: float = Field(..., description="Toplam işlem süresi")
    enhanced_content: List[EnhancedContent] = Field(..., description="Geliştirilmiş içerikler") 