"""
Modül 4: Gelişmiş Metin Nitelikleri Analizi

Bu modül, Modül 3'ten gelen makale metinlerinin dilsel ve anlamsal niteliklerini 
daha derinlemesine analiz eder. Odak noktamız, metnin duygu tonu, tarafsızlığı, 
okunabilirlik düzeyi ve dilin karmaşıklığı gibi objektif metrikleri çıkarmaktır.

Algoritma Akışı:
1. Makale Verilerinin Alınması (Modül 3'ten)
2. Gelişmiş Metin Niteliklerinin Çıkarılması
   - Duygu Analizi (Hugging Face Transformers)
   - Tarafsızlık/Subjektiflik Analizi
   - Okunabilirlik Skorları (TextStat)
   - Temel Metin Metrikleri
3. Çıkarılan Niteliklerin Yapılandırılması
4. Nihai Makale Verilerinin Hazırlanması
"""

from .routes import router as module4_router

__all__ = ["module4_router"] 