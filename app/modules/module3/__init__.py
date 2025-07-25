"""
Modül 3: Akıllı İçerik Analizi ve Alaka Düzeyi Kontrolü

Bu modül, taranmış web sayfalarının içeriğini ayrıştırır, temizler ve 
kullanıcının prompt'una olan anlamsal alaka düzeyini belirler.

Algoritma:
1. Ham HTML içeriğinin alınması
2. Ana içerik ayrıştırma ve metin temizleme (python-readability + BeautifulSoup4)
3. Metin ön işleme (Hugging Face Transformers için hazırlık)
4. Prompt ve makale arasında anlamsal alaka düzeyi tespiti (Sentence-BERT)
5. Filtreleme ve ön eleme
6. Meta veri ve içerik hazırlığı
"""

from .routes import router as module3_router

__all__ = ["module3_router"] 