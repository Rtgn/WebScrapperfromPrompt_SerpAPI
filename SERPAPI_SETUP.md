# SerpApi Kurulum ve Yapılandırma Rehberi

## 🔑 SerpApi Nedir?

SerpApi, Google, Bing, DuckDuckGo gibi arama motorlarından programatik olarak sonuç almanızı sağlayan bir API servisidir. Bu API sayesinde:

- **Gerçek arama sonuçları** alabilirsiniz
- **Bot tespiti** sorunları yaşamazsınız
- **Yapılandırılmış veri** elde edersiniz
- **Zaman filtreleme** yapabilirsiniz

## 🚀 SerpApi Hesabı Oluşturma

### 1. SerpApi'ye Kayıt Olun
```bash
# SerpApi ana sayfasına gidin
https://serpapi.com/
```

### 2. Ücretsiz Plan Seçin
- **100 arama/ay** ücretsiz
- **Tüm arama motorları** desteklenir
- **API anahtarı** hemen verilir

### 3. API Anahtarınızı Alın
- Dashboard'da **API Key** bölümüne gidin
- API anahtarınızı kopyalayın

## ⚙️ Projeye SerpApi Entegrasyonu

### 1. Bağımlılıkları Yükleyin
```bash
pip install google-search-results==2.4.2
```

### 2. Ortam Değişkeni Ayarlayın

#### Windows (.env dosyası)
```env
SERPAPI_KEY=your_actual_api_key_here
```

#### macOS/Linux (.env dosyası)
```env
SERPAPI_KEY=your_actual_api_key_here
```

### 3. .env Dosyası Oluşturun
```bash
# Proje ana dizininde .env dosyası oluşturun
cp env.example .env

# .env dosyasını düzenleyin
# SERPAPI_KEY=your_actual_api_key_here satırını gerçek API anahtarınızla değiştirin
```

## 🧪 SerpApi Test Etme

### 1. Basit Test
```python
import os
from serpapi import GoogleSearch

# API anahtarınızı ayarlayın
api_key = "your_actual_api_key_here"

# Test araması yapın
search = GoogleSearch({
    "q": "artificial intelligence ethics",
    "api_key": api_key,
    "num": 5
})

results = search.get_dict()

# Sonuçları kontrol edin
if "organic_results" in results:
    print(f"Found {len(results['organic_results'])} results")
    for result in results['organic_results']:
        print(f"- {result.get('title', 'No title')}")
        print(f"  URL: {result.get('link', 'No URL')}")
```

### 2. API Test Scripti
```bash
# test_serpapi.py dosyası oluşturun
python test_serpapi.py
```

## 📊 SerpApi Özellikleri

### Desteklenen Arama Motorları
- **Google** (varsayılan)
- **Bing**
- **DuckDuckGo**
- **Yahoo**
- **Yandex**

### Zaman Filtreleme
```python
# Son 24 saat
search_params["tbs"] = "qdr:d"

# Son hafta
search_params["tbs"] = "qdr:w"

# Son ay
search_params["tbs"] = "qdr:m"

# Son 3 ay
search_params["tbs"] = "qdr:3m"

# Son yıl
search_params["tbs"] = "qdr:y"

# Özel tarih aralığı
search_params["tbs"] = "cdr:1,cd_min:01/01/2024,cd_max:12/31/2024"
```

### Dil ve Ülke Ayarları
```python
search_params = {
    "q": "artificial intelligence",
    "hl": "en",      # Dil (en, tr, de, fr, vb.)
    "gl": "us",      # Ülke (us, tr, de, fr, vb.)
    "api_key": api_key
}
```

## 💰 Fiyatlandırma

### Ücretsiz Plan
- **100 arama/ay**
- Tüm özellikler
- API anahtarı hemen

### Ücretli Planlar
- **Starter**: $50/ay - 5,000 arama
- **Professional**: $100/ay - 15,000 arama
- **Enterprise**: Özel fiyatlandırma

### Kullanım Takibi
- Dashboard'da **Usage** bölümünden takip edebilirsiniz
- API anahtarınızın kullanım istatistiklerini görebilirsiniz

## 🔧 Gelişmiş Yapılandırma

### 1. Rate Limiting
```python
import asyncio

# Arama istekleri arasında bekleme
await asyncio.sleep(1.0)  # 1 saniye bekle
```

### 2. Hata Yönetimi
```python
try:
    search = GoogleSearch(search_params)
    results = search.get_dict()
except Exception as e:
    print(f"SerpApi error: {e}")
    # Fallback to simulation
```

### 3. Sonuç Sayısı Sınırı
```python
# Maksimum 100 sonuç (SerpApi limiti)
search_params["num"] = min(max_results, 100)
```

## 🚨 Sorun Giderme

### Yaygın Sorunlar

#### 1. API Anahtarı Hatası
```bash
# .env dosyasını kontrol edin
cat .env | grep SERPAPI_KEY

# API anahtarının doğru olduğundan emin olun
```

#### 2. Rate Limiting
```bash
# Arama istekleri arasında daha uzun bekleme süresi ekleyin
await asyncio.sleep(2.0)  # 2 saniye bekle
```

#### 3. Quota Aşımı
```bash
# Dashboard'da kullanım istatistiklerini kontrol edin
# Yeni ay başında quota sıfırlanır
```

#### 4. Ağ Bağlantısı
```bash
# İnternet bağlantınızı kontrol edin
# Firewall ayarlarını kontrol edin
```

## 📈 Performans İpuçları

### 1. Önbellekleme
```python
# Aynı sorguları tekrar yapmayın
# Sonuçları veritabanında saklayın
```

### 2. Akıllı Sorgu Oluşturma
```python
# Çok genel sorgulardan kaçının
# Spesifik anahtar kelimeler kullanın
```

### 3. Batch İşlemler
```python
# Birden fazla sorguyu tek seferde yapın
# Rate limiting'e dikkat edin
```

## 🔗 Faydalı Linkler

- [SerpApi Ana Sayfa](https://serpapi.com/)
- [SerpApi Dokümantasyonu](https://serpapi.com/docs)
- [Python Kütüphanesi](https://github.com/serpapi/google-search-results-python)
- [API Playground](https://serpapi.com/playground)

## 📞 Destek

SerpApi ile ilgili sorunlar için:
1. [SerpApi Dokümantasyonu](https://serpapi.com/docs)
2. [GitHub Issues](https://github.com/serpapi/google-search-results-python/issues)
3. [SerpApi Support](https://serpapi.com/support)

---

🎉 **SerpApi entegrasyonu tamamlandı!** Artık gerçek arama sonuçları alabilirsiniz. 