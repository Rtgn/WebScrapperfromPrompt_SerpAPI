#!/usr/bin/env python3
"""
Environment Setup Script
========================

Bu script .env dosyasını oluşturmanıza yardımcı olur.
"""

import os

def create_env_file():
    """Create .env file with template"""
    
    env_content = """# SerpApi Configuration
# Gerçek SerpApi anahtar kelimenizi buraya ekleyin
SERPAPI_KEY=your_serpapi_key_here

# Database Configuration
DATABASE_URL=sqlite:///discovery.db

# Redis Configuration (opsiyonel)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Application Settings
DEBUG=false
LOG_LEVEL=INFO

# Module 1 Settings
SPACY_MODEL=en_core_web_sm
KEYBERT_MODEL=all-MiniLM-L6-v2
MAX_KEYWORDS=20
SIMILARITY_THRESHOLD=0.7
EXPANSION_THRESHOLD=0.8

# Module 2 Settings
DEFAULT_CRAWL_DEPTH=2
DEFAULT_MAX_PAGES=100
REQUEST_TIMEOUT=30
"""
    
    try:
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        print("✅ .env dosyası oluşturuldu!")
        print("📝 Şimdi .env dosyasını düzenleyin ve SERPAPI_KEY değerini ekleyin:")
        print("   1. .env dosyasını açın")
        print("   2. SERPAPI_KEY=your_serpapi_key_here satırını bulun")
        print("   3. your_serpapi_key_here yerine gerçek anahtar kelimenizi yazın")
        print("   4. Dosyayı kaydedin")
        print()
        print("🔑 SerpApi anahtar kelimesi almak için:")
        print("   https://serpapi.com/ adresine gidin")
        print("   Ücretsiz hesap oluşturun")
        print("   API anahtar kelimenizi alın")
        
        return True
        
    except Exception as e:
        print(f"❌ .env dosyası oluşturulamadı: {e}")
        return False

def check_env_file():
    """Check if .env file exists and has SerpApi key"""
    
    if not os.path.exists('.env'):
        print("❌ .env dosyası bulunamadı!")
        return False
    
    try:
        with open('.env', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'SERPAPI_KEY=' in content:
            lines = content.split('\n')
            for line in lines:
                if line.startswith('SERPAPI_KEY='):
                    key = line.split('=')[1].strip()
                    if key and key != 'your_serpapi_key_here':
                        print(f"✅ SerpApi anahtar kelimesi bulundu: {key[:10]}...")
                        return True
                    else:
                        print("⚠️  SerpApi anahtar kelimesi henüz ayarlanmamış")
                        return False
        
        print("❌ SERPAPI_KEY satırı .env dosyasında bulunamadı")
        return False
        
    except Exception as e:
        print(f"❌ .env dosyası okunamadı: {e}")
        return False

def main():
    """Main function"""
    print("🔧 Environment Setup Script")
    print("=" * 40)
    
    # Check if .env exists
    if os.path.exists('.env'):
        print("📁 .env dosyası mevcut, kontrol ediliyor...")
        if check_env_file():
            print("✅ SerpApi anahtar kelimesi doğru ayarlanmış!")
            return
        else:
            print("⚠️  SerpApi anahtar kelimesi eksik veya yanlış")
    else:
        print("📁 .env dosyası bulunamadı, oluşturuluyor...")
        if not create_env_file():
            return
    
    print("\n📋 Manuel Kurulum Talimatları:")
    print("1. .env dosyasını metin editöründe açın")
    print("2. SERPAPI_KEY=your_serpapi_key_here satırını bulun")
    print("3. your_serpapi_key_here yerine gerçek anahtar kelimenizi yazın")
    print("4. Dosyayı kaydedin")
    print("5. Server'ı yeniden başlatın")
    print()
    print("🔑 SerpApi anahtar kelimesi almak için:")
    print("   https://serpapi.com/ adresine gidin")
    print("   Ücretsiz hesap oluşturun")
    print("   API anahtar kelimenizi alın")

if __name__ == "__main__":
    main() 