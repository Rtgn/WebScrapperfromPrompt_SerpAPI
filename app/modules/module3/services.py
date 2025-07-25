"""
Modül 3: Akıllı İçerik Analizi ve Alaka Düzeyi Kontrolü - Servisler

README'deki algoritma akışına göre implementasyon:
1. Ham HTML İçeriğinin Alınması
2. Ana İçerik Ayrıştırma ve Metin Temizleme (python-readability + BeautifulSoup4)
3. Metin Ön İşleme (Hugging Face Transformers için hazırlık)
4. Prompt ve Makale Arasında Anlamsal Alaka Düzeyi Tespiti (Sentence-BERT)
5. Filtreleme ve Ön Eleme
6. Meta Veri ve İçerik Hazırlığı
"""

import asyncio
import aiohttp
import time
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
from datetime import datetime, timedelta
import json

# HTML parsing ve temizleme
from bs4 import BeautifulSoup
from readability import Document

# NLP ve embedding
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

# Dil tespiti
from langdetect import detect, LangDetectException

# Modül 1 taklidi için
import spacy
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class KeywordExtractor:
    """Modül 1 taklidi: Anahtar kelime çıkarma"""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.keybert = KeyBERT()
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    
    def extract_keywords(self, prompt: str, max_keywords: int = 20) -> Dict[str, Any]:
        """Prompt'tan anahtar kelime çıkarır (Modül 1 taklidi)"""
        start_time = time.time()
        
        try:
            # SpaCy ile NER ve POS tabanlı çıkarma
            doc = self.nlp(prompt)
            
            # NER tabanlı anahtar kelimeler
            ner_keywords = [ent.text.lower() for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT']]
            
            # POS tabanlı anahtar kelimeler (NOUN, PROPN, ADJ)
            pos_keywords = [token.text.lower() for token in doc if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and not token.is_stop]
            
            # KeyBERT ile semantic anahtar kelimeler
            keybert_keywords = self.keybert.extract_keywords(prompt, keyphrase_ngram_range=(1, 2), stop_words='english')
            semantic_keywords = [kw[0].lower() for kw in keybert_keywords[:10]]
            
            # TF-IDF tabanlı anahtar kelimeler
            tfidf_keywords = self._extract_tfidf_keywords(prompt)
            
            # Tüm anahtar kelimeleri birleştir ve sırala
            all_keywords = list(set(ner_keywords + pos_keywords + semantic_keywords + tfidf_keywords))
            
            # En önemli anahtar kelimeleri seç
            selected_keywords = all_keywords[:max_keywords]
            
            # Sonuç formatı
            extracted_keywords = []
            for i, keyword in enumerate(selected_keywords):
                extracted_keywords.append({
                    'keyword': keyword,
                    'rank': i + 1,
                    'score': 1.0 - (i * 0.05),  # Basit skor hesaplama
                    'source': 'combined'
                })
            
            extraction_time = time.time() - start_time
            
            return {
                'prompt': prompt,
                'extracted_keywords': extracted_keywords,
                'total_keywords': len(extracted_keywords),
                'extraction_time': extraction_time,
                'methods_used': ['spacy_ner', 'spacy_pos', 'keybert', 'tfidf']
            }
            
        except Exception as e:
            logger.error(f"Anahtar kelime çıkarma hatası: {str(e)}")
            # Fallback: basit kelime çıkarma
            words = prompt.lower().split()
            words = [w for w in words if len(w) > 3 and w.isalpha()]
            fallback_keywords = words[:max_keywords]
            
            return {
                'prompt': prompt,
                'extracted_keywords': [{'keyword': kw, 'rank': i+1, 'score': 0.5, 'source': 'fallback'} 
                                     for i, kw in enumerate(fallback_keywords)],
                'total_keywords': len(fallback_keywords),
                'extraction_time': time.time() - start_time,
                'methods_used': ['fallback'],
                'error': str(e)
            }
    
    def _extract_tfidf_keywords(self, text: str) -> List[str]:
        """TF-IDF tabanlı anahtar kelime çıkarma"""
        try:
            # Basit TF-IDF hesaplama
            words = text.lower().split()
            word_freq = {}
            for word in words:
                if word.isalpha() and len(word) > 3:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # En sık geçen kelimeleri döndür
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, freq in sorted_words[:10]]
        except:
            return []


class URLGenerator:
    """Modül 2 entegrasyonu: Gerçek SERP API ile URL oluşturma"""
    
    def __init__(self):
        self.discovery_service = None
    
    async def _get_discovery_service(self):
        """Modül 2'nin discovery servisini al"""
        if self.discovery_service is None:
            from app.modules.module2.services import discovery_service
            # Servisin initialize edilip edilmediğini kontrol et
            if not discovery_service.initialized:
                await discovery_service.initialize()
            self.discovery_service = discovery_service
        return self.discovery_service
    
    async def generate_urls(self, keywords: List[str], max_urls: int = 50, 
                          time_limit_days: int = 30, language: str = "en", 
                          country: str = "us") -> Dict[str, Any]:
        """Anahtar kelimelerden gerçek URL'ler oluşturur (Modül 2 entegrasyonu)"""
        start_time = time.time()
        
        try:
            # Modül 2'nin discovery servisini al
            discovery_service = await self._get_discovery_service()
            
            # Modül 2'nin SearchRequest şemasını kullan
            from app.modules.module2.schemas import SearchRequest, TimeFilter
            
            # Time filter'ı ayarla
            time_filter = None
            if time_limit_days:
                if time_limit_days <= 1:
                    time_filter = TimeFilter.PAST_24_HOURS
                elif time_limit_days <= 7:
                    time_filter = TimeFilter.PAST_WEEK
                elif time_limit_days <= 30:
                    time_filter = TimeFilter.PAST_MONTH
                elif time_limit_days <= 90:
                    time_filter = TimeFilter.PAST_3_MONTHS
                elif time_limit_days <= 365:
                    time_filter = TimeFilter.PAST_YEAR
            
            # SearchRequest oluştur
            search_request = SearchRequest(
                keywords=keywords,
                max_results=max_urls,
                time_filter=time_filter,
                search_engine="google",
                language=language,
                country=country
            )
            
            logger.info(f"Modül 2 ile arama başlatılıyor: {len(keywords)} anahtar kelime, {max_urls} sonuç")
            
            # Modül 2'nin gerçek search_sources metodunu çağır
            search_response = await discovery_service.search_sources(search_request)
            
            # URL'leri çıkar
            generated_urls = [result.url for result in search_response.search_results]
            
            generation_time = time.time() - start_time
            
            logger.info(f"Modül 2 arama tamamlandı: {len(generated_urls)} URL bulundu")
            
            return {
                'keywords': keywords,
                'generated_urls': generated_urls,
                'total_urls': len(generated_urls),
                'generation_time': generation_time,
                'search_query': " ".join(keywords[:5]),
                'search_stats': search_response.search_stats,
                'processing_time': search_response.processing_time
            }
            
        except Exception as e:
            logger.error(f"Modül 2 URL oluşturma hatası: {str(e)}")
            # Fallback: basit URL listesi
            fallback_urls = [
                "https://example.com/article1",
                "https://techcrunch.com/ai-ethics-transparency",
                "https://wired.com/artificial-intelligence-ethics"
            ]
            
            return {
                'keywords': keywords,
                'generated_urls': fallback_urls[:max_urls],
                'total_urls': len(fallback_urls[:max_urls]),
                'generation_time': time.time() - start_time,
                'search_query': " ".join(keywords[:5]),
                'error': str(e),
                'fallback_used': True
            }


class ContentExtractor:
    """İçerik çıkarma ve temizleme"""
    
    def __init__(self):
        self.session = None
    
    async def _get_session(self):
        """HTTP session oluştur"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def extract_content(self, url: str) -> Optional[Dict[str, Any]]:
        """URL'den içerik çıkarır (README algoritma adım 1-2)"""
        start_time = time.time()
        
        try:
            session = await self._get_session()
            
            # 1. Ham HTML İçeriğinin Alınması
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # SSL doğrulamasını devre dışı bırak (geliştirme için)
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as temp_session:
                async with temp_session.get(url, headers=headers, allow_redirects=True) as response:
                    if response.status not in [200, 201, 202]:
                        logger.warning(f"URL erişilemedi: {url}, Status: {response.status}")
                        return None
                    
                    html_content = await response.text()
            
            # 2. Ana İçerik Ayrıştırma ve Metin Temizleme
            # python-readability kullanımı
            doc = Document(html_content)
            
            # Ana içeriği çıkar
            main_content = doc.summary()
            title = doc.title() or "[no-title]"
            
            # BeautifulSoup ile ek temizlik
            soup = BeautifulSoup(main_content, 'html.parser')
            
            # Metin temizleme
            text_content = self._clean_text(soup.get_text())
            
            # Meta verileri çıkar
            meta_soup = BeautifulSoup(html_content, 'html.parser')
            author = self._extract_author(meta_soup)
            publish_date = self._extract_publish_date(meta_soup)
            language = self._detect_language(text_content)
            
            # Özet oluştur
            summary = self._create_summary(text_content)
            
            # Kelime sayısı
            word_count = len(text_content.split())
            
            processing_time = time.time() - start_time
            
            return {
                'url': url,
                'title': title,
                'content': text_content,
                'summary': summary,
                'author': author,
                'publish_date': publish_date,
                'language': language,
                'word_count': word_count,
                'extraction_method': 'readability',
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"İçerik çıkarma hatası ({url}): {str(e)}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Metin temizleme"""
        # Birden fazla boşlukları tek boşluğa indirme
        text = re.sub(r'\s+', ' ', text)
        # Özel karakterleri temizleme
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        # Başındaki ve sonundaki boşlukları kaldırma
        text = text.strip()
        return text
    
    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        """Yazar bilgisini çıkar"""
        # Yaygın yazar meta tag'leri
        author_selectors = [
            'meta[name="author"]',
            'meta[property="article:author"]',
            '.author',
            '.byline',
            '[class*="author"]'
        ]
        
        for selector in author_selectors:
            element = soup.select_one(selector)
            if element:
                return element.get('content') or element.get_text().strip()
        
        return None
    
    def _extract_publish_date(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Yayın tarihini çıkar"""
        # Yaygın tarih meta tag'leri
        date_selectors = [
            'meta[name="publish_date"]',
            'meta[property="article:published_time"]',
            'meta[name="date"]',
            'time[datetime]'
        ]
        
        for selector in date_selectors:
            element = soup.select_one(selector)
            if element:
                date_str = element.get('content') or element.get('datetime')
                if date_str:
                    try:
                        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    except:
                        continue
        
        return None
    
    def _detect_language(self, text: str) -> Optional[str]:
        """Dil tespiti"""
        try:
            return detect(text[:1000])  # İlk 1000 karakteri kullan
        except LangDetectException:
            return None
    
    def _create_summary(self, text: str, max_sentences: int = 3) -> str:
        """Metin özeti oluştur"""
        sentences = text.split('.')
        if len(sentences) <= max_sentences:
            return text
        
        summary_sentences = sentences[:max_sentences]
        return '. '.join(summary_sentences) + '.'


class RelevanceAnalyzer:
    """Anlamsal alaka düzeyi analizi"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def calculate_relevance_score(self, prompt: str, content: str) -> float:
        """Prompt ve içerik arasında anlamsal alaka düzeyi hesaplar (README algoritma adım 4)"""
        try:
            # Embedding oluşturma
            prompt_embedding = self.model.encode(prompt, convert_to_tensor=True)
            content_embedding = self.model.encode(content[:1000], convert_to_tensor=True)  # İlk 1000 karakter
            
            # Kosinüs benzerliği hesaplama
            cosine_sim = torch.nn.functional.cosine_similarity(
                prompt_embedding.unsqueeze(0), 
                content_embedding.unsqueeze(0)
            )
            
            return float(cosine_sim.item())
            
        except Exception as e:
            logger.error(f"Alaka düzeyi hesaplama hatası: {str(e)}")
            return 0.0
    
    def is_within_time_limit(self, publish_date: Optional[datetime], time_limit_days: int) -> bool:
        """Zaman sınırı kontrolü"""
        if not publish_date or not time_limit_days:
            return True
        
        try:
            # Tarih formatını normalize et
            if publish_date.tzinfo is not None:
                # Timezone-aware datetime'i naive'a çevir
                publish_date = publish_date.replace(tzinfo=None)
            
            cutoff_date = datetime.now().replace(tzinfo=None) - timedelta(days=time_limit_days)
            is_within = publish_date >= cutoff_date
            
            logger.info(f"    - Yayın tarihi: {publish_date}")
            logger.info(f"    - Kesme tarihi: {cutoff_date}")
            logger.info(f"    - Zaman sınırı içinde: {is_within}")
            
            return is_within
        except Exception as e:
            logger.warning(f"Tarih karşılaştırma hatası: {e}")
            return True  # Hata durumunda kabul et


class ContentFilter:
    """İçerik filtreleme"""
    
    def filter_content(self, content: Dict[str, Any], 
                      relevance_threshold: float,
                      excluded_domains: List[str],
                      required_keywords: List[str],
                      language: str) -> bool:
        """İçeriği filtreler (README algoritma adım 5)"""
        
        # Debug bilgisi
        relevance_score = content.get('relevance_score', 0)
        logger.info(f"  Filtreleme kontrolü:")
        logger.info(f"    - Alaka skoru: {relevance_score:.4f} vs eşik: {relevance_threshold}")
        logger.info(f"    - Alaka düzeyi geçiyor mu: {relevance_score >= relevance_threshold}")
        
        # Alaka düzeyi kontrolü
        if relevance_score < relevance_threshold:
            logger.info(f"    ❌ Alaka düzeyi düşük")
            return False
        
        # Domain kontrolü
        url = content.get('url', '')
        domain = urlparse(url).netloc
        if domain in excluded_domains:
            logger.info(f"    ❌ Domain hariç tutulmuş: {domain}")
            return False
        
        # Zorunlu anahtar kelime kontrolü
        if required_keywords:
            content_text = content.get('content', '').lower()
            if not any(keyword.lower() in content_text for keyword in required_keywords):
                logger.info(f"    ❌ Zorunlu anahtar kelimeler bulunamadı")
                return False
        
        # Dil filtresi kaldırıldı
        # if language and content.get('language') != language:
        #     logger.info(f"    ❌ Dil uyumsuz: {content.get('language')} vs {language}")
        #     return False
        
        logger.info(f"    ✅ Tüm filtreler geçildi")
        return True


class ContentAnalysisService:
    """Ana içerik analizi servisi"""
    
    def __init__(self):
        self.keyword_extractor = KeywordExtractor()
        self.url_generator = URLGenerator()
        self.content_extractor = ContentExtractor()
        self.relevance_analyzer = RelevanceAnalyzer()
        self.content_filter = ContentFilter()
    
    async def analyze_content_from_prompt(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Ana analiz fonksiyonu - README algoritmasının tamamını uygular"""
        start_time = time.time()
        
        try:
            # Parametreleri al
            prompt = request['prompt']
            max_keywords = request.get('max_keywords', 20)
            max_urls = request.get('max_urls', 50)
            time_limit_days = request.get('time_limit_days', 30)
            relevance_threshold = request.get('relevance_threshold', 0.6)
            excluded_domains = request.get('excluded_domains', [])
            required_keywords = request.get('required_keywords', [])
            language = request.get('language', 'en')
            
            logger.info(f"İçerik analizi başlatılıyor: {prompt[:100]}...")
            
            # Modül 1 taklidi: Anahtar kelime çıkarma
            logger.info("Modül 1: Anahtar kelime çıkarma başlatılıyor...")
            module1_results = self.keyword_extractor.extract_keywords(prompt, max_keywords)
            keywords = [kw['keyword'] for kw in module1_results['extracted_keywords']]
            logger.info(f"Modül 1 tamamlandı. {len(keywords)} anahtar kelime çıkarıldı.")
            
            # Modül 2 entegrasyonu: Gerçek URL oluşturma
            logger.info("Modül 2: Gerçek SERP API ile URL oluşturma başlatılıyor...")
            module2_results = await self.url_generator.generate_urls(
                keywords=keywords, 
                max_urls=max_urls,
                time_limit_days=time_limit_days,
                language=language,
                country="us"
            )
            urls = module2_results['generated_urls']
            logger.info(f"Modül 2 tamamlandı. {len(urls)} gerçek URL bulundu.")
            logger.info(f"Modül 2 sonuçları: {module2_results}")
            
            # URL'lerin boş olup olmadığını kontrol et
            if not urls:
                logger.error("Modül 2'den URL gelmedi! Fallback kullanılacak.")
                # Fallback URL'ler
                urls = [
                    "https://example.com/article1",
                    "https://techcrunch.com/ai-ethics-transparency",
                    "https://wired.com/artificial-intelligence-ethics"
                ]
                module2_results['generated_urls'] = urls
                module2_results['total_urls'] = len(urls)
                module2_results['fallback_used'] = True
            
            # Modül 3: İçerik analizi
            logger.info("Modül 3: İçerik analizi başlatılıyor...")
            relevant_content = []
            errors = []
            
            # URL'leri paralel olarak işle
            tasks = [self._analyze_single_url(url, prompt, time_limit_days, relevance_threshold, 
                                            excluded_domains, required_keywords, language) 
                    for url in urls]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    errors.append(str(result))
                elif result and result.get('is_relevant'):
                    relevant_content.append(result)
            
            # İstatistikler
            total_processing_time = time.time() - start_time
            average_relevance = sum(r['relevance_score'] for r in relevant_content) / len(relevant_content) if relevant_content else 0.0
            
            # Modül 3 sonuçları
            module3_results = {
                'total_urls_processed': len(urls),
                'relevant_articles': len(relevant_content),
                'average_relevance_score': average_relevance,
                'processing_time': total_processing_time,
                'errors': errors
            }
            
            logger.info(f"Modül 3 tamamlandı. {len(relevant_content)} alakalı makale bulundu.")
            
            return {
                'module1_results': module1_results,
                'module2_results': module2_results,
                'module3_results': module3_results,
                'total_urls_processed': len(urls),
                'relevant_articles': len(relevant_content),
                'average_relevance_score': average_relevance,
                'total_processing_time': total_processing_time,
                'relevant_content': relevant_content,
                'errors': errors
            }
            
        except Exception as e:
            logger.error(f"İçerik analizi hatası: {str(e)}")
            return {
                'module1_results': {},
                'module2_results': {},
                'module3_results': {},
                'total_urls_processed': 0,
                'relevant_articles': 0,
                'average_relevance_score': 0.0,
                'total_processing_time': time.time() - start_time,
                'relevant_content': [],
                'errors': [str(e)]
            }
    
    async def _analyze_single_url(self, url: str, prompt: str, time_limit_days: int,
                                relevance_threshold: float, excluded_domains: List[str],
                                required_keywords: List[str], language: str) -> Optional[Dict[str, Any]]:
        """Tek bir URL'yi analiz eder"""
        try:
            # İçerik çıkar
            content_data = await self.content_extractor.extract_content(url)
            if not content_data:
                return None
            
            # Alaka düzeyi hesapla
            relevance_score = self.relevance_analyzer.calculate_relevance_score(prompt, content_data['content'])
            
            # Zaman sınırı kontrolü
            is_within_time_limit = self.relevance_analyzer.is_within_time_limit(
                content_data['publish_date'], time_limit_days
            )
            
            # Analiz sonucu
            analysis_result = {
                'content': content_data,
                'relevance_score': relevance_score,
                'is_relevant': relevance_score >= relevance_threshold,
                'is_within_time_limit': is_within_time_limit,
                'analysis_time': content_data['processing_time']
            }
            
            # Debug bilgisi ekle
            logger.info(f"URL analiz edildi: {url}")
            logger.info(f"  - Alaka skoru: {relevance_score:.4f}")
            logger.info(f"  - Eşik: {relevance_threshold}")
            logger.info(f"  - Alakalı mı: {relevance_score >= relevance_threshold}")
            logger.info(f"  - Zaman sınırı içinde mi: {is_within_time_limit}")
            
            # Filtreleme
            if self.content_filter.filter_content(
                analysis_result, relevance_threshold, excluded_domains, 
                required_keywords, language
            ):
                logger.info(f"  ✅ URL kabul edildi: {url}")
                return analysis_result
            else:
                logger.info(f"  ❌ URL reddedildi: {url}")
                return None
            
        except Exception as e:
            logger.error(f"URL analiz hatası ({url}): {str(e)}")
            return None
    
    async def close(self):
        """Kaynakları temizle"""
        if self.content_extractor.session:
            await self.content_extractor.session.close() 