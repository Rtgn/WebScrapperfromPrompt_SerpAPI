"""
Modül 4: Gelişmiş Metin Nitelikleri Analizi - Servisler

Bu modül, Modül 1, 2, 3'ü zincirleme çalıştırır ve sonuçlara metin nitelikleri analizi ekler.
"""

import asyncio
import time
import logging
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

# NLP ve metin analizi
import spacy
from transformers import pipeline
import textstat
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

# Modül importları
from app.modules.module1.services import keyword_extractor
from app.modules.module2.services import discovery_service
from app.modules.module3.services import ContentAnalysisService

# Şemalar
from .schemas import (
    SentimentType, ObjectivityType, SentimentAnalysis, ObjectivityAnalysis,
    ReadabilityScores, TextMetrics, TextQualityAnalysis, EnhancedContent,
    Module1Result, Module2Result, Module3Result, TextQualityResponse
)

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Duygu analizi servisi"""
    
    def __init__(self):
        """Duygu analizi modellerini yükle"""
        try:
            # Hugging Face duygu analizi pipeline'ı
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            
            # NLTK sentiment analyzer (yedek)
            try:
                nltk.data.find('vader_lexicon')
            except LookupError:
                nltk.download('vader_lexicon')
            
            self.vader_analyzer = SentimentIntensityAnalyzer()
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Duygu analizi modelleri yüklenemedi: {e}")
            self.initialized = False
    
    def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """Metnin duygu analizini yap"""
        if not self.initialized:
            return self._fallback_sentiment_analysis(text)
        
        try:
            # Hugging Face ile duygu analizi
            result = self.sentiment_pipeline(text[:512])[0]  # Model limiti
            
            # Skorları normalize et
            if result['label'] == 'POSITIVE':
                sentiment = SentimentType.POSITIVE
                positive_score = result['score']
                negative_score = 1 - result['score']
                neutral_score = 0.0
            elif result['label'] == 'NEGATIVE':
                sentiment = SentimentType.NEGATIVE
                positive_score = 1 - result['score']
                negative_score = result['score']
                neutral_score = 0.0
            else:
                sentiment = SentimentType.NEUTRAL
                positive_score = 0.0
                negative_score = 0.0
                neutral_score = result['score']
            
            return SentimentAnalysis(
                sentiment=sentiment,
                confidence=result['score'],
                positive_score=positive_score,
                negative_score=negative_score,
                neutral_score=neutral_score
            )
            
        except Exception as e:
            logger.error(f"Duygu analizi hatası: {e}")
            return self._fallback_sentiment_analysis(text)
    
    def _fallback_sentiment_analysis(self, text: str) -> SentimentAnalysis:
        """NLTK VADER ile yedek duygu analizi"""
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            
            if scores['compound'] >= 0.05:
                sentiment = SentimentType.POSITIVE
                confidence = abs(scores['compound'])
            elif scores['compound'] <= -0.05:
                sentiment = SentimentType.NEGATIVE
                confidence = abs(scores['compound'])
            else:
                sentiment = SentimentType.NEUTRAL
                confidence = 0.5
            
            return SentimentAnalysis(
                sentiment=sentiment,
                confidence=confidence,
                positive_score=scores['pos'],
                negative_score=scores['neg'],
                neutral_score=scores['neu']
            )
            
        except Exception as e:
            logger.error(f"Yedek duygu analizi hatası: {e}")
            # Varsayılan nötr sonuç
            return SentimentAnalysis(
                sentiment=SentimentType.NEUTRAL,
                confidence=0.5,
                positive_score=0.33,
                negative_score=0.33,
                neutral_score=0.34
            )


class ObjectivityAnalyzer:
    """Tarafsızlık/subjektiflik analizi servisi"""
    
    def __init__(self):
        """Tarafsızlık analizi için gerekli bileşenleri yükle"""
        try:
            # Subjektif kelime listeleri
            self.subjective_words = self._load_subjective_words()
            self.objective_words = self._load_objective_words()
            
            # SpaCy modeli
            self.nlp = spacy.load("en_core_web_sm")
            
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Tarafsızlık analizi modelleri yüklenemedi: {e}")
            self.initialized = False
    
    def analyze_objectivity(self, text: str) -> ObjectivityAnalysis:
        """Metnin tarafsızlık analizini yap"""
        if not self.initialized:
            return self._fallback_objectivity_analysis(text)
        
        try:
            # Metni tokenize et
            doc = self.nlp(text.lower())
            words = [token.text for token in doc if token.is_alpha and not token.is_stop]
            
            # Subjektif ve objektif kelime sayılarını hesapla
            subjective_count = sum(1 for word in words if word in self.subjective_words)
            objective_count = sum(1 for word in words if word in self.objective_words)
            total_words = len(words)
            
            if total_words == 0:
                return self._fallback_objectivity_analysis(text)
            
            # Skorları hesapla
            subjectivity_score = subjective_count / total_words
            objectivity_score = objective_count / total_words
            
            # Tarafsızlık türünü belirle
            if objectivity_score > 0.6:
                objectivity_type = ObjectivityType.OBJECTIVE
                confidence = objectivity_score
            elif subjectivity_score > 0.4:
                objectivity_type = ObjectivityType.SUBJECTIVE
                confidence = subjectivity_score
            else:
                objectivity_type = ObjectivityType.MIXED
                confidence = 0.5
            
            return ObjectivityAnalysis(
                objectivity_type=objectivity_type,
                objectivity_score=objectivity_score,
                subjectivity_score=subjectivity_score,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Tarafsızlık analizi hatası: {e}")
            return self._fallback_objectivity_analysis(text)
    
    def _fallback_objectivity_analysis(self, text: str) -> ObjectivityAnalysis:
        """Basit yedek tarafsızlık analizi"""
        # Basit kelime sayımı
        subjective_indicators = ['i think', 'i believe', 'in my opinion', 'personally', 'feel', 'feelings']
        objective_indicators = ['research', 'study', 'data', 'evidence', 'fact', 'analysis']
        
        text_lower = text.lower()
        subjective_count = sum(1 for indicator in subjective_indicators if indicator in text_lower)
        objective_count = sum(1 for indicator in objective_indicators if indicator in text_lower)
        
        if objective_count > subjective_count:
            objectivity_type = ObjectivityType.OBJECTIVE
            objectivity_score = 0.7
            subjectivity_score = 0.3
        elif subjective_count > objective_count:
            objectivity_type = ObjectivityType.SUBJECTIVE
            objectivity_score = 0.3
            subjectivity_score = 0.7
        else:
            objectivity_type = ObjectivityType.MIXED
            objectivity_score = 0.5
            subjectivity_score = 0.5
        
        return ObjectivityAnalysis(
            objectivity_type=objectivity_type,
            objectivity_score=objectivity_score,
            subjectivity_score=subjectivity_score,
            confidence=0.6
        )
    
    def _load_subjective_words(self) -> set:
        """Subjektif kelime listesini yükle"""
        return {
            'think', 'believe', 'feel', 'opinion', 'personally', 'personally', 'guess',
            'suppose', 'assume', 'hope', 'wish', 'want', 'like', 'love', 'hate',
            'dislike', 'prefer', 'favorite', 'best', 'worst', 'amazing', 'terrible',
            'wonderful', 'awful', 'beautiful', 'ugly', 'interesting', 'boring',
            'exciting', 'dull', 'surprising', 'shocking', 'disappointing', 'satisfying'
        }
    
    def _load_objective_words(self) -> set:
        """Objektif kelime listesini yükle"""
        return {
            'research', 'study', 'data', 'evidence', 'fact', 'analysis', 'report',
            'findings', 'results', 'conclusion', 'method', 'procedure', 'technique',
            'measurement', 'observation', 'experiment', 'test', 'examination',
            'investigation', 'survey', 'statistics', 'percentage', 'number', 'figure',
            'table', 'chart', 'graph', 'diagram', 'model', 'theory', 'hypothesis'
        }


class ReadabilityAnalyzer:
    """Okunabilirlik analizi servisi"""
    
    def __init__(self):
        """TextStat kütüphanesini kullan"""
        self.initialized = True
    
    def analyze_readability(self, text: str) -> ReadabilityScores:
        """Metnin okunabilirlik skorlarını hesapla"""
        try:
            return ReadabilityScores(
                flesch_reading_ease=textstat.flesch_reading_ease(text),
                flesch_kincaid_grade=textstat.flesch_kincaid_grade(text),
                gunning_fog=textstat.gunning_fog(text),
                smog_index=textstat.smog_index(text),
                coleman_liau_index=textstat.coleman_liau_index(text),
                automated_readability_index=textstat.automated_readability_index(text),
                dale_chall_score=textstat.dale_chall_readability_score(text),
                linsear_write_formula=textstat.linsear_write_formula(text),
                fernandez_huerta=textstat.fernandez_huerta(text),
                szigriszt_pazos=textstat.szigriszt_pazos(text),
                gutierrez_polini=textstat.gutierrez_polini(text),
                crawford=textstat.crawford(text)
            )
            
        except Exception as e:
            logger.error(f"Okunabilirlik analizi hatası: {e}")
            # Varsayılan skorlar
            return ReadabilityScores(
                flesch_reading_ease=50.0,
                flesch_kincaid_grade=10.0,
                gunning_fog=12.0,
                smog_index=8.0,
                coleman_liau_index=10.0,
                automated_readability_index=10.0,
                dale_chall_score=8.0,
                linsear_write_formula=10.0,
                fernandez_huerta=50.0,
                szigriszt_pazos=50.0,
                gutierrez_polini=50.0,
                crawford=10.0
            )


class TextMetricsAnalyzer:
    """Temel metin metrikleri analizi servisi"""
    
    def __init__(self):
        """SpaCy modelini yükle"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.initialized = True
        except Exception as e:
            logger.error(f"Text metrics analyzer yüklenemedi: {e}")
            self.initialized = False
    
    def analyze_text_metrics(self, text: str) -> TextMetrics:
        """Temel metin metriklerini hesapla"""
        if not self.initialized:
            return self._fallback_text_metrics(text)
        
        try:
            doc = self.nlp(text)
            
            # Temel sayımlar
            word_count = len([token for token in doc if token.is_alpha])
            sentence_count = len(list(doc.sents))
            character_count = len(text)
            paragraph_count = len(text.split('\n\n'))
            
            # Ortalama hesaplamaları
            average_word_length = sum(len(token.text) for token in doc if token.is_alpha) / max(word_count, 1)
            average_sentence_length = word_count / max(sentence_count, 1)
            
            # Benzersiz kelimeler
            unique_words = len(set(token.text.lower() for token in doc if token.is_alpha))
            vocabulary_diversity = unique_words / max(word_count, 1)
            
            return TextMetrics(
                word_count=word_count,
                sentence_count=sentence_count,
                character_count=character_count,
                paragraph_count=paragraph_count,
                average_word_length=round(average_word_length, 2),
                average_sentence_length=round(average_sentence_length, 2),
                unique_words=unique_words,
                vocabulary_diversity=round(vocabulary_diversity, 3)
            )
            
        except Exception as e:
            logger.error(f"Text metrics analizi hatası: {e}")
            return self._fallback_text_metrics(text)
    
    def _fallback_text_metrics(self, text: str) -> TextMetrics:
        """Basit yedek metin metrikleri"""
        words = text.split()
        sentences = text.split('.')
        paragraphs = text.split('\n\n')
        
        word_count = len(words)
        sentence_count = len([s for s in sentences if s.strip()])
        character_count = len(text)
        paragraph_count = len([p for p in paragraphs if p.strip()])
        
        average_word_length = sum(len(word) for word in words) / max(word_count, 1)
        average_sentence_length = word_count / max(sentence_count, 1)
        
        unique_words = len(set(word.lower() for word in words))
        vocabulary_diversity = unique_words / max(word_count, 1)
        
        return TextMetrics(
            word_count=word_count,
            sentence_count=sentence_count,
            character_count=character_count,
            paragraph_count=paragraph_count,
            average_word_length=round(average_word_length, 2),
            average_sentence_length=round(average_sentence_length, 2),
            unique_words=unique_words,
            vocabulary_diversity=round(vocabulary_diversity, 3)
        )


class TextQualityService:
    """Ana metin nitelikleri analizi servisi"""
    
    def __init__(self):
        """Tüm analiz servislerini başlat"""
        self.sentiment_analyzer = SentimentAnalyzer()
        self.objectivity_analyzer = ObjectivityAnalyzer()
        self.readability_analyzer = ReadabilityAnalyzer()
        self.text_metrics_analyzer = TextMetricsAnalyzer()
        
        # Modül 3 servisi
        self.content_analysis_service = ContentAnalysisService()
    
    async def analyze_text_quality_from_prompt(self, request: Dict[str, Any]) -> TextQualityResponse:
        """Prompt'tan başlayarak tüm modülleri zincirleme çalıştır ve metin nitelikleri analizi ekle"""
        start_time = time.time()
        
        try:
            logger.info("🚀 Modül 4: Metin nitelikleri analizi başlatılıyor...")
            
            # Modül 1: Anahtar kelime çıkarma
            logger.info("📝 Modül 1: Anahtar kelime çıkarma...")
            module1_start = time.time()
            try:
                module1_result = keyword_extractor.extract_keywords(
                    prompt=request['prompt'],
                    max_keywords=request.get('max_keywords', 15)
                )
                module1_time = time.time() - module1_start
                logger.info(f"✅ Modül 1 tamamlandı: {module1_result['total_keywords']} anahtar kelime, {module1_time:.2f}s")
            except Exception as e:
                logger.error(f"❌ Modül 1 hatası: {str(e)}")
                raise
            
            # Modül 2: URL keşfi
            logger.info("🔍 Modül 2: URL keşfi...")
            module2_start = time.time()
            
            try:
                # Modül 2'yi başlat
                await discovery_service.initialize()
                
                # Anahtar kelimeleri al
                keywords = [kw['keyword'] for kw in module1_result['extracted_keywords']]
                logger.info(f"🔑 Anahtar kelimeler: {keywords[:5]}...")  # İlk 5'ini göster
                
                # URL'leri keşfet
                from app.modules.module2.schemas import SearchRequest, TimeFilter
                search_request = SearchRequest(
                    prompt=request['prompt'],
                    keywords=keywords,
                    max_results=request.get('max_urls', 20),
                    time_filter=TimeFilter.PAST_MONTH if request.get('time_limit_days', 30) <= 30 else TimeFilter.PAST_YEAR,
                    language=request.get('language', 'en')
                )
                
                module2_result = await discovery_service.search_sources(search_request)
                module2_time = time.time() - module2_start
                logger.info(f"✅ Modül 2 tamamlandı: {len(module2_result.search_results)} URL, {module2_time:.2f}s")
            except Exception as e:
                logger.error(f"❌ Modül 2 hatası: {str(e)}")
                raise
            
            # Modül 2 sonuçlarından URL'leri al
            generated_urls = [result.url for result in module2_result.search_results]
            
            # Modül 3: İçerik analizi
            logger.info("📄 Modül 3: İçerik analizi...")
            module3_start = time.time()
            
            try:
                # Modül 3'ü çağır
                module3_request = {
                    'prompt': request['prompt'],
                    'max_keywords': request.get('max_keywords', 15),
                    'max_urls': request.get('max_urls', 20),
                    'time_limit_days': request.get('time_limit_days', 30),
                    'relevance_threshold': request.get('relevance_threshold', 0.1),
                    'max_content_length': request.get('max_content_length', 2048),
                    'excluded_domains': request.get('excluded_domains', []),
                    'required_keywords': request.get('required_keywords', []),
                    'language': request.get('language', 'en')
                }
                
                module3_result = await self.content_analysis_service.analyze_content_from_prompt(module3_request)
                module3_time = time.time() - module3_start
                logger.info(f"✅ Modül 3 tamamlandı: {len(module3_result.get('relevant_content', []))} alakalı içerik, {module3_time:.2f}s")
            except Exception as e:
                logger.error(f"❌ Modül 3 hatası: {str(e)}")
                raise
            
            # Modül 4: Metin nitelikleri analizi
            logger.info("🔬 Modül 4: Metin nitelikleri analizi...")
            module4_start = time.time()
            
            enhanced_content = []
            for content_item in module3_result.get('relevant_content', []):
                try:
                    # Metin nitelikleri analizi yap
                    text_quality = await self._analyze_text_quality(content_item['content']['content'])
                    
                    # Geliştirilmiş içerik oluştur
                    enhanced_content.append(EnhancedContent(
                        url=content_item['content']['url'],
                        title=content_item['content']['title'],
                        content=content_item['content']['content'],
                        summary=content_item['content']['summary'],
                        author=content_item['content']['author'],
                        publish_date=content_item['content']['publish_date'],
                        language=content_item['content']['language'],
                        word_count=content_item['content']['word_count'],
                        extraction_method=content_item['content']['extraction_method'],
                        processing_time=content_item['content']['processing_time'],
                        relevance_score=content_item['relevance_score'],
                        is_relevant=content_item['is_relevant'],
                        is_within_time_limit=content_item['is_within_time_limit'],
                        text_quality_analysis=text_quality
                    ))
                except Exception as e:
                    logger.error(f"İçerik analizi hatası: {e}")
                    continue
            
            module4_time = time.time() - module4_start
            total_time = time.time() - start_time
            
            # Sonuçları hazırla
            return TextQualityResponse(
                module1_results=Module1Result(
                    prompt=request['prompt'],  # Orijinal request'ten al
                    extracted_keywords=module1_result['extracted_keywords'],
                    total_keywords=module1_result['total_keywords'],
                    extraction_time=module1_time
                ),
                module2_results=Module2Result(
                    keywords=keywords,
                    generated_urls=generated_urls,
                    total_urls=len(generated_urls),
                    generation_time=module2_time
                ),
                module3_results=Module3Result(
                    total_urls_processed=module3_result['total_urls_processed'],
                    relevant_articles=module3_result['relevant_articles'],
                    average_relevance_score=module3_result['average_relevance_score'],
                    processing_time=module3_time,
                    errors=module3_result.get('errors', [])
                ),
                module4_results={
                    'total_articles_analyzed': len(enhanced_content),
                    'analysis_time': module4_time,
                    'average_sentiment_confidence': sum(
                        c.text_quality_analysis.sentiment_analysis.confidence 
                        for c in enhanced_content
                    ) / max(len(enhanced_content), 1),
                    'average_readability_score': sum(
                        c.text_quality_analysis.readability_scores.flesch_reading_ease 
                        for c in enhanced_content
                    ) / max(len(enhanced_content), 1)
                },
                total_urls_processed=module3_result['total_urls_processed'],
                relevant_articles=len(enhanced_content),
                average_relevance_score=module3_result['average_relevance_score'],
                total_processing_time=total_time,
                enhanced_content=enhanced_content
            )
            
        except Exception as e:
            logger.error(f"Metin nitelikleri analizi hatası: {e}")
            logger.error(f"Hata detayı: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    async def _analyze_text_quality(self, text: str) -> TextQualityAnalysis:
        """Tek bir metin için tüm nitelik analizlerini yap"""
        analysis_start = time.time()
        
        # Duygu analizi
        sentiment_analysis = self.sentiment_analyzer.analyze_sentiment(text)
        
        # Tarafsızlık analizi
        objectivity_analysis = self.objectivity_analyzer.analyze_objectivity(text)
        
        # Okunabilirlik analizi
        readability_scores = self.readability_analyzer.analyze_readability(text)
        
        # Metin metrikleri
        text_metrics = self.text_metrics_analyzer.analyze_text_metrics(text)
        
        analysis_time = time.time() - analysis_start
        
        return TextQualityAnalysis(
            sentiment_analysis=sentiment_analysis,
            objectivity_analysis=objectivity_analysis,
            readability_scores=readability_scores,
            text_metrics=text_metrics,
            analysis_time=analysis_time
        )
    
    async def close(self):
        """Servisi kapat"""
        await self.content_analysis_service.close()


# Global servis instance'ı
text_quality_service = TextQualityService() 