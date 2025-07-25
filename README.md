# Advanced Research API

Bu proje, gelişmiş doğal dil işleme, anahtar kelime çıkarımı ve dinamik kaynak keşfi için modüler bir FastAPI uygulamasıdır.

## Proje Yapısı


## Modül 1: Gelişmiş Anahtar Kelime Çıkarımı ve Anlamsal Genişletme

### 1.1 Teknoloji Seçimi ve Rolleri

#### SpaCy
**Rolü:** Prompt'un hızlı ve verimli bir şekilde temel metin ön işlemden geçirilmesi (tokenizasyon, lemmatizasyon, durdurma kelimesi filtreleme) ve özellikle varlık tanıma (Named Entity Recognition - NER). Bu sayede, prompt içindeki somut anahtar terimler ve özel isimler (örneğin şirket adları, ürünler, önemli kavramlar) belirlenecek.

**Neden:** Üretim ortamları için optimize edilmiş hızı, gelişmiş NER yetenekleri ve sağlam temel NLP işlevleri nedeniyle tercih edilir.

#### KeyBERT
**Rolü:** SpaCy tarafından ön işlenmiş prompt metninden, bağlamsal ve anlamsal olarak en alakalı anahtar kelimeleri ve anahtar ifadeleri çıkarmak. Bu, prompt'un genel anlamını ve vurguladığı konuları derinlemesine anlamayı sağlar.

**Neden:** BERT tabanlı güçlü bir model kullanarak kelimelerin sadece sözdizimsel değil, anlamsal ilişkilerini de anlayarak çok daha alakalı ve kaliteli anahtar kelimeler üretir. Tek kelimelerin yanı sıra, çok kelimeli ifadeleri ("yeni modeller", "ses getiren gelişmeler") de yakalayabilir.

#### Hugging Face Transformers (özellikle Sentence-BERT modelleri)
**Rolü:** KeyBERT'ten çıkan anahtar kelimelerin anlamsal vektör temsillerini (embedding'ler) oluşturmak ve bu embedding'leri kullanarak daha geniş bir kavram dağarcığı içindeki anlamsal olarak benzer kelimeleri ve kavramları genişletmek.

**Neden:** Sentence-BERT modelleri, cümleler ve kelimeler arası anlamsal benzerlikleri yüksek doğrulukla ölçmek için özel olarak eğitilmiştir. Bu sayede, "yapay zeka"dan "makine öğrenimi" gibi ilgili ama direkt prompt'ta geçmeyen terimleri türetebiliriz.

#### Scikit-learn (veya NumPy)
**Rolü:** Vektörler arasındaki kosinüs benzerliğini hesaplayarak anlamsal alaka düzeyini belirlemek.

**Neden:** Vektör tabanlı matematiksel işlemler için standart ve verimli bir kütüphanedir.

### 1.2 Algoritma Akışı (Sistematik ve Gelişmiş Adımlar)

İşte bu teknolojilerin entegre ve akıllıca çalışacağı adım adım algoritma:

#### 1. Prompt Girdisi Alımı
- API, kullanıcıdan string formatında bir serbest metin prompt'u alır.
- **Örnek Prompt:** "Yapay zeka alanındaki son gelişmeler, çığır açan yeni modeller, startup inovasyonları ve bulgular hakkında detaylı bilgi."

#### 2. Temel Metin Ön İşleme ve Varlık Tanıma (SpaCy ile)
- **SpaCy modelini yükle:** Genellikle `en_core_web_dm` gibi bir İngilizce model yüklenir.
- **Prompt'u İşle:** Gelen prompt metni SpaCy işleme hattına verilir (`nlp(prompt)`).
- **Tokenizasyon ve Lemmatizasyon:** Metin kelimelere/token'lara ayrılır ve her kelimenin kök formu (lemma) bulunur. Bu, "gelişmeler" kelimesinin "gelişme" olarak işlem görmesini sağlar.
- **Durdurma Kelimesi Filtreleme:** "Ve", "hakkında", "bir" gibi anlamsız kelimeler (stop words) filtrelenir.
- **Named Entity Recognition (NER):** SpaCy'nin NER özelliği kullanılarak prompt'taki özel isimler, kurumlar, ürünler ve anahtar kavramlar (örn. ORG, PRODUCT, CONCEPT) tespit edilir.
- **İlk Anahtar Kelime Havuzu Oluşturma:** NER tarafından tespit edilen direkt varlıklar, **"Çekirdek Anahtar Kelime Havuzu"**muzun ilk ve yüksek öncelikli parçası olarak kaydedilir.

#### 3. Anlamsal Anahtar Kelime ve Anahtar İfade Çıkarımı (KeyBERT ile)
- **KeyBERT modelini yükle:** `all-MiniLM-L6-v2` gibi Sentence-BERT tabanlı bir model kullanılır.
- **Anahtar Kelime Çıkarımı:** SpaCy tarafından ön işlenmiş prompt metni (veya orijinal prompt'un anlamını koruyan temiz bir versiyonu) KeyBERT'e girdi olarak verilir.
- **Parametre Ayarı:** `nr_candidates` (aday kelime öbeği sayısı), `top_n` (dönülecek en iyi kelime öbeği sayısı) ve `diversity` (çeşitlilik) gibi parametreler optimize edilir.
- **Örnek Çıktı:** "yapay zeka", "yeni modeller", "startup inovasyonları", "çığır açan gelişmeler", "bulgular".
- **"Çekirdek Anahtar Kelime Havuzu"na Ekleme:** KeyBERT'ten gelen bu anlamsal anahtar kelimeler ve anahtar ifadeler, **"Çekirdek Anahtar Kelime Havuzu"**muza eklenir.

#### 4. Anlamsal Genişletme ve Konsept Haritalama (Hugging Face Transformers ve Scikit-learn ile)
- **Embedding Modelini Yükle:** Genellikle KeyBERT'in de kullandığı aynı Sentence-BERT modelini (`all-MiniLM-L6-v2` gibi) doğrudan Hugging Face Transformers üzerinden yükleriz.
- **Geniş Kavram Dağarcığı:** Önceden hazırlanmış veya dinamik olarak oluşturulmuş geniş bir kelime/kavram dağarcığı bulunur (örneğin, Wikipedia makale başlıkları, akademik terimler listeleri, genel terimler sözlüğü). Bu dağarcıktaki her bir kelime/kavramın embedding'leri önceden hesaplanmış ve hızlı erişilebilir bir veritabanında (örn. vektör veritabanı veya bellekteki sözlük) saklanmalıdır. Bu, her seferinde yeniden hesaplama maliyetini düşürür.
- **Anlamsal Genişletme Döngüsü:**
  - **"Çekirdek Anahtar Kelime Havuzu"**ndaki her bir anahtar kelime/ifade için:
    - Bu anahtar kelimenin/ifadenin embedding'i (vektör temsili) Sentence-BERT modeli kullanılarak hesaplanır.
    - Hesaplanan bu embedding, geniş kavram dağarcığındaki diğer tüm kelime/kavram embedding'leri ile kosinüs benzerliği açısından karşılaştırılır (Scikit-learn'den `cosine_similarity`).
    - Belirli bir yüksek benzerlik eşiğinin (örneğin 0.7 veya 0.8) üzerindeki kelimeler/kavramlar, anlamsal olarak ilgili kabul edilir.
    - Bu yeni, ilgili kelimeler ve kavramlar **"Genişletilmiş Anahtar Kelime Havuzu"**na eklenir.

**Örnek Genişletmeler:**
- "yapay zeka" → "makine öğrenimi", "derin öğrenme", "sinir ağları", "algoritmalar", "AI", "veri bilimi"
- "startup inovasyonları" → "fintech girişimleri", "biyoteknoloji startupları", "teknoloji inovasyonu", "girişimcilik ekosistemi"

#### 5. Nihai Arama Sorguları Listesinin Oluşturulması ve Optimizasyonu
- **"Çekirdek Anahtar Kelime Havuzu"** ve **"Genişletilmiş Anahtar Kelime Havuzu"** birleştirilir.
- Tekrarlayanlar ve çok yakın anlamsal kelimeler ayıklanır (isteğe bağlı olarak tekrar kosinüs benzerliği kontrolü ile).
- Bu birleşik havuzdaki anahtar kelimelerden ve ifadelerden, arama motorlarında kullanılmak üzere çeşitli kombinasyonlarda ve varyasyonlarda arama sorguları dinamik olarak oluşturulur:
  - **Tek kelime sorguları:** "yapay zeka", "makine öğrenimi"
  - **Çok kelimeli ifadeler (tırnak içinde):** "yeni modeller", "startup inovasyonları"
  - **Kombinasyonlar (AND/OR mantığıyla):** "yapay zeka" AND "yeni modeller", "startup" OR "girişim"
  - **Uzun kuyruk sorguları:** Birden fazla anahtar kelimeyi içeren daha spesifik sorgular.

Bu aşamanın çıktısı, bir sonraki modül olan "Dinamik Kaynak Keşfi ve Genişletilmiş Arama" için hazır, çeşitlendirilmiş ve zengin bir arama sorguları listesi olacaktır.

## Modül 2: Dinamik Kaynak Keşfi ve Genişletilmiş Arama

### 2.1 Teknoloji Seçimi ve Rolleri

Bu modül için, yüksek performans, esneklik ve bot tespitini aşma yetenekleri sunan teknolojileri seçeceğiz.

#### Arama Motoru Sonuç Sayfası (SERP) Veri Kaynağı
**Teknoloji:** Google Custom Search API veya SerpApi / Bright Data SERP API.

**Rolü:** Anahtar kelime havuzumuzdaki sorguları kullanarak programatik olarak arama motoru sonuçlarını (URL'ler, başlıklar, kısa açıklamalar) almak. Bu, tarama için başlangıç "tohum" URL'lerini sağlayacak.

**Neden:** Doğrudan arama motoru sayfalarını (google.com, bing.com vb.) scrape etmek, IP yasakları, CAPTCHA'lar ve sürekli değişen HTML yapıları nedeniyle oldukça zordur. Bu tür API'ler, bu zorlukları bizim için yönetir ve temiz, yapılandırılmış veri sunar. Şimdilik SerpApi kullanalım, ileriki aşamalarda CSE yedek olarak entegre edilebilir.

#### Web Tarama (Crawling) Çerçevesi
**Teknoloji:** Scrapy.

**Rolü:** Arama motoru sonuçlarından elde edilen URL'leri ve daha sonra bu sayfalardan keşfedilen iç/dış linkleri sistematik, ölçeklenebilir ve verimli bir şekilde taramak. Dinamik olarak linkleri takip etme, hata yönetimi ve proxy rotasyonu gibi özellikleri yönetmek.

**Neden:** Scrapy, büyük ölçekli ve karmaşık web tarama projeleri için tasarlanmış tam teşekküllü bir Python çerçevesidir. Asenkron yapısı sayesinde aynı anda birden fazla isteği işleyebilir, genişletilebilir boru hatları (pipelines) ve ara yazılımlar (middlewares) ile özelleştirilebilir, böylece bot tespiti mekanizmalarını daha iyi aşabiliriz.

#### Dinamik İçerik Yükleme (Opsiyonel ama Önemli)
**Teknoloji:** Scrapy-Splash (Scrapy ile entegre) veya Selenium (Scrapy dışında ayrı bir süreç olarak).

**Rolü:** Eğer taranan web siteleri, JavaScript kullanarak içeriği dinamik olarak yüklüyorsa (yani, sayfanın kaynak HTML'sinde görünmüyorsa), bu araçlar bir web tarayıcısı motorunu taklit ederek JavaScript'i çalıştırır ve tam yüklü HTML'yi elde etmemizi sağlar.

**Neden:** Modern web sitelerinin çoğu dinamik içerik kullanır. Bu araçlar olmadan birçok bilgiye erişemeyebiliriz. Scrapy-Splash, Scrapy ile daha entegre bir çözüm sunarken, Selenium daha genel bir tarayıcı otomasyon aracıdır.

#### Proxy ve User-Agent Yönetimi
**Teknoloji:** Proxy Havuzları (ticari veya kendi kurulan) ve User-Agent rotasyonu için Scrapy Middlewares.

**Rolü:** Web sitelerinin IP bazlı yasaklamalarını veya bot tespitini önlemek. Her istekte farklı bir IP adresi veya tarayıcı kimliği kullanarak insan benzeri bir davranış sergilemek.

**Neden:** Sürekli tarama, web sunucuları tarafından kolayca tespit edilebilir ve IP adresinizin kara listeye alınmasına neden olabilir. Bu, tarama sürecinin kesintiye uğramaması için hayati öneme sahiptir.

#### Veri Depolama (Geçici)
**Teknoloji:** Redis (kuyruk yönetimi için) ve PostgreSQL/MongoDB (toplanan URL'lerin ve meta verilerin geçici/kalıcı depolanması için).

**Rolü:** Taranacak URL'leri yönetmek (kuyruk sistemi), taranan URL'leri kaydetmek ve duplicate (tekrar eden) taramaları önlemek.

### 2.2 Algoritma Akışı (Sistematik ve Akıllıca)

İşte anahtar kelime havuzumuzu kullanarak dinamik kaynak keşfi ve genişletilmiş arama sürecinin adımları:

#### 1. Arama Sorgularının Oluşturulması (Modül 1 Çıktısı)
- Modül 1'den gelen çeşitlendirilmiş ve zengin arama sorguları listesi alınır. Bu liste, arama motorlarında kullanılacak anahtar kelime kombinasyonlarını içerir.
- **Örnek Sorgular:** "yapay zeka yeni modeller", "AI startup inovasyonları", "derin öğrenme çığır açan gelişmeler", "yapay zeka etiği bulgular"

#### 2. Başlangıç Tohum URL'lerinin Elde Edilmesi (SERP API ile)
- Oluşturulan her bir arama sorgusu, Google Custom Search API veya SerpApi gibi bir arama motoru API'sine gönderilir.
- **Zaman Filtresi Entegrasyonu:** Eğer kullanılan SERP API zaman aralığı filtresi sunuyorsa (ki çoğu sunar), anahtar kelime sorgusuna ek olarak bu zaman filtresini de göndeririz. Örneğin, "past month" (son ay) veya belirli bir tarih aralığı (date_published:2024-06-23..2024-07-23). Bu sayede, arama motoru zaten yalnızca belirtilen zaman aralığına ait sonuçları döndürür.
- **Örnek SERP API Sorgusu:** `api.search(query="Yapay zeka etiği", date_range="last_30_days")`
- API'den dönen ilk N adet (örneğin 50-100 adet) arama sonucu URL'si ve ilgili meta bilgileri (başlık, özet) toplanır.
- Bu URL'ler, tarama sürecimizin ilk "tohum" listesini oluşturur ve bir veritabanına (örneğin PostgreSQL) kaydedilir.
- **Akıllı Ön-Süzgeç (İsteğe Bağlı ama Önemli):** Bu aşamada, toplanan URL'lerin alan adları üzerinde basit bir itibar/otorite kontrolü yapılabilir (örneğin, önceden belirlenmiş güvenilir domain listesi veya alan adı otorite API'leri ile). Güvenilirlik puanı düşük olan veya alakasız görünen domainler erken aşamada elenebilir.

#### 3. Dinamik Web Taramasının Başlatılması (Scrapy ile)
- Toplanan "tohum" URL'leri, Scrapy Spider'ına gönderilir.
- **Scrapy Middlewares (Proxy ve User-Agent Rotasyonu):** İstekler gönderilirken otomatik olarak farklı proxy sunucuları ve user-agent'lar kullanılır.
- **robots.txt Kontrolü:** Her yeni alan adına istek göndermeden önce, Scrapy otomatik olarak ilgili sitenin robots.txt dosyasını kontrol eder ve taranmasına izin verilmeyen yolları atlar.
- **HTML İndirme ve Ayrıştırma:** Scrapy, web sayfalarının HTML içeriğini indirir. Eğer sayfa dinamik içerik kullanıyorsa, Scrapy-Splash entegrasyonu devreye girer ve JavaScript'i çalıştırarak tam HTML'yi elde eder.

#### 4. Akıllı Link Keşfi ve Prioritizasyonu
- Her taranan sayfadan tüm iç ve dış linkler ayrıştırılır.
- **Link Filtreleme:**
  - Görüntü, video, PDF gibi dosya uzantılarına sahip linkler (eğer metin tabanlı makale aramıyorsak) atlanır.
  - Sosyal medya paylaşım linkleri, yorum linkleri gibi alakasız linkler filtrelenir.
  - Daha önce taranmış veya taranmakta olan linkler (duplicate control) atlanır.
  - **Zaman Filtresi (İkincil Kontrol):** Scrapy spider'ı, bir URL'yi taramadan önce, mümkünse URL'den veya hedef sayfanın tahmini yayın tarihinden (eğer URL yapısında varsa) bir ön kontrol yapabilir. Eğer çok bariz bir şekilde eski bir tarihse (örn. 2010), bu linkin taranması atlanabilir. Bu bir tür "erken çıkış" mekanizmasıdır.
- **Link Prioritizasyonu (Akıllı Yönlendirme):** Bu, modülün en akıllı kısımlarından biridir.
  - **URL ve Anchor Text Analizi:** Linkin URL'sinde veya bağlantı metninde (anchor text) Modül 1'den gelen anahtar kelime havuzundaki terimlerin geçip geçmediği kontrol edilir.
  - **Anlamsal Benzerlik (İsteğe Bağlı ama Güçlü):** Linkin metninin veya URL'sinin, prompt'umuza ve anahtar kelime havuzumuza anlamsal olarak ne kadar yakın olduğu (yine Sentence-BERT embedding'leri ile kosinüs benzerliği hesaplanarak) belirlenir. Daha yüksek benzerlik puanına sahip linklere daha yüksek tarama önceliği verilir.
  - **Alan Adı İtibarı:** Bağlantı verilen alan adının ön-süzgeç aşamasında belirlenen itibarı veya potansiyel güvenilirliği dikkate alınır.
  - **Tarama Derinliği Kontrolü:** Sonsuz tarama döngülerine girmemek için belirli bir derinlikten sonra (örneğin, tohum URL'sinden 3-4 tıklama derinliği) link takibi durdurulur veya önceliği düşürülür.
- **Yeni URL'leri Scrapy Kuyruğuna Ekleme:** Önceliklendirilmiş ve filtrelenmiş linkler, Scrapy'nin tarama kuyruğuna (scheduler) eklenir.

#### 5. Potansiyel Makale Tespiti ve Meta Veri Çıkarımı
- Taranan her web sayfasında, makale başlığı, ana metin, yayın tarihi, yazar gibi bilgilerin bulunduğu potansiyel alanlar tespit edilir (CSS seçicileri veya XPath kullanılarak).
- Bu aşamada sadece metin çekilir, detaylı içerik analizi ve güvenilirlik değerlendirmesi (Modül 3 ve Modül 4) sonraki adımlarda yapılacaktır.
- Toplanan her makale için temel meta veriler (URL, başlık, olası özet/giriş paragrafı) çıkarılır ve geçici olarak depolanır (örneğin, bir veritabanı veya Scrapy'nin item pipeline'ı aracılığıyla).

Bu modülün çıktısı, Modül 1'den gelen anahtar kelime havuzuna anlamsal olarak yüksek düzeyde alakalı ve taranmış web sayfalarının (veya potansiyel makalelerin) yapılandırılmış bir listesi olacaktır. Bu liste, bir sonraki aşamada detaylı içerik analizi ve güvenilirlik değerlendirmesi için hazırdır.

## Modül 3: Akıllı İçerik Analizi ve Alaka Düzeyi Kontrolü

Bu modülün temel amacı, Modül 2'den gelen taranmış web sayfalarının içeriğini ayrıştırmak, temizlemek ve bu içeriğin kullanıcının başlangıç prompt'una ne kadar anlamsal olarak alakalı olduğunu belirleyen bir skor veya sınıflandırma atamaktır.

### 3.1 Teknoloji Seçimi ve Rolleri

Bu modül için, metin işleme ve anlamsal analizde en gelişmiş araçları kullanacağız.

#### Metin Ayrıştırma ve Temizleme
**Teknoloji:** BeautifulSoup4 (genellikle Scrapy ile entegre veya ayrı olarak kullanılır) ve python-readability.

**Rolü:** HTML yapısından sadece ana makale metnini (başlık, paragraf, resim açıklamaları vb.) çıkarmak, navigasyon menüleri, reklamlar, dipnotlar gibi alakasız veya gürültülü elementleri temizlemek. python-readability özellikle web sayfasındaki "okunabilir" ana içeriği belirlemede çok etkilidir.

**Neden:** Temiz ve odaklanmış metin, sonraki NLP adımlarının doğruluğu için hayati öneme sahiptir.

#### Gelişmiş Metin İşleme ve Anlamsal Analiz
**Teknoloji:** Hugging Face Transformers Kütüphanesi (özellikle Sentence-BERT veya benzeri anlamsal embedding modelleri) ve PyTorch.

**Rolü:** Temizlenmiş makale metnini (veya makalenin özetini/ilk paragrafını) ve orijinal prompt'u anlamsal vektör temsillerine (embedding'lere, yani PyTorch tensörlerine) dönüştürmek. Bu embedding'ler arasındaki benzerliği hesaplayarak makalenin prompt'a olan anlamsal alaka düzeyini belirlemek. Ayrıca, zero-shot sınıflandırma veya topik modelleme gibi daha ileri tekniklerle makalenin doğrudan prompt'un belirttiği konu kategorisine girip girmediğini tespit etmek.

**Neden:** Modern Transformer modelleri, kelimelerin ve cümlelerin sadece yüzeydeki benzerliklerini değil, derinlemesine anlamsal ilişkilerini de anlayabilir. Bu, anahtar kelime eşleştirmesinden çok daha doğru ve "akıllı" bir alaka düzeyi belirleme sağlar. PyTorch entegrasyonu, performans ve ekosistem uyumu sağlar.

### 3.2 Algoritma Akışı (Sistematik ve Akıllıca)

İşte taranan her web sayfası için uygulanacak akış:

#### 1. Ham HTML İçeriğinin Alınması
- Modül 2'den gelen taranmış her bir web sayfasının ham HTML içeriği alınır.

#### 2. Ana İçerik Ayrıştırma ve Metin Temizleme
- **python-readability kullanımı:** Sayfanın ana makale metni (başlık, ana paragraf içeriği, yazar bilgisi, yayın tarihi gibi kritik meta verilerle birlikte) çıkarılır. Reklamlar, menüler, footer gibi alakasız öğeler elenir.
- **Zaman Filtresi (Doğrulama Noktası):** Makalenin yayın tarihi (genellikle datetime formatında) çıkarılır çıkarılmaz, bu tarih ile güncel tarih arasında bir kontrol yapılır. Eğer makalenin yayın tarihi, API isteğinde belirtilen zaman aralığının dışındaysa, bu makale hemen işlem hattından çıkarılır ve bir sonraki modüle geçirilmez.
- **Filtreleme Mantığı:** Bu kontrol, makalenin alaka düzeyi veya güvenilirliği hesaplanmadan önce yapılır, böylece gereksiz hesaplama yükünden kaçınılır.
- **Ek temizlik:** Çıkarılan metin üzerinde ek temizlik yapılır: Birden fazla boşlukları tek boşluğa indirme, özel karakterleri kaldırma, standart kodlama formatına dönüştürme.
- **BeautifulSoup4 ile ek kontrol:** Eğer python-readability yeterli olmazsa veya ek kontrol gerekirse, BeautifulSoup4 ile spesifik CSS seçicileri veya XPath ifadeleri kullanılarak ana makale içeriği (genellikle article etiketi veya id="main-content" gibi yaygın yapılar) çekilebilir.

#### 3. Metin Ön İşleme (Hugging Face Transformers için Hazırlık)
- Temizlenmiş makale metni, Hugging Face Transformers modelinin beklentilerine uygun hale getirilir. Genellikle bu, metni belirli bir uzunluğa kırpmak (modelin maksimum token limitine uymak için, özellikle çok uzun makaleler için özet veya ilk paragrafları kullanmak) ve tokenizasyon için hazıra getirmektir.

#### 4. Prompt ve Makale Arasında Anlamsal Alaka Düzeyi Tespiti

**Embedding Oluşturma:**
- Orijinal prompt metninin (Modül 1'den gelen) anlamsal vektör temsili (PyTorch tensörü) Sentence-BERT modeli (örneğin all-MiniLM-L6-v2) kullanılarak hesaplanır.
- Temizlenmiş makale metninin (veya makale metninin belirgin bir kısmının, örn. ilk 512 token) anlamsal vektör temsili (PyTorch tensörü) aynı Sentence-BERT modeli kullanılarak hesaplanır.

**Kosinüs Benzerliği Hesaplama:**
- Prompt'un embedding tensörü ile makale metninin embedding tensörü arasında kosinüs benzerliği doğrudan `torch.nn.functional.cosine_similarity` kullanılarak hesaplanır. Bu, 0 ile 1 arasında bir değer döner, 1'e ne kadar yakınsa, anlamsal olarak o kadar benzerdirler.
- Bu değer, makalenin prompt'a olan "Alaka Düzeyi Skoru" olarak kaydedilir.

#### 5. Filtreleme ve Ön Eleme
- Hesaplanan "Alaka Düzeyi Skoru" belirli bir eşik değerinin (örneğin 0.6 veya 0.7) altında olan makaleler elenir. Bu, sonraki aşamalara (güvenilirlik değerlendirmesi) sadece gerçekten ilgili makalelerin geçmesini sağlar.

#### 6. Meta Veri ve İçerik Hazırlığı
Alaka düzeyi testini geçen makaleler için:
- **Makale Başlığı:** Çıkarılır.
- **Ana Metin:** Temizlenmiş ana metin depolanır.
- **Özet/Giriş Paragrafı:** Uzun makaleler için, ana metnin ilk birkaç cümlesi veya otomatik bir özetleyici (eğer kullanılacaksa) özet olarak çıkarılır.
- **Yayın Tarihi, Yazar (varsa):** Çekilir.
- **URL:** Orjinal URL kaydedilir.
- **Alaka Düzeyi Skoru:** Hesaplanan skor eklenir.

Bu modülün çıktısı, yüksek alaka düzeyi skoruna sahip, temizlenmiş ve yapılandırılmış makale verilerinin bir koleksiyonu olacaktır. Bu veriler, bir sonraki modülde güvenilirlik değerlendirmesi ve kalite kontrolü için hazır olacaktır.


## Modül 4: Gelişmiş Metin Nitelikleri Analizi (Güncellenmiş Kapsam)

Bu modül, Modül 3'ten gelen makale metinlerinin dilsel ve anlamsal niteliklerini daha derinlemesine analiz etmeyi hedefler. Odak noktamız, metnin duygu tonu, tarafsızlığı, okunabilirlik düzeyi ve dilin karmaşıklığı gibi objektif metrikleri çıkarmaktır. Bu bilgiler, kullanıcının aldığı sonucun genel karakterini anlamasına yardımcı olacak ek bağlamsal veri sağlayacaktır.

### 4.1 Teknoloji Seçimi ve Rolleri (Ücretsiz Bileşenlerle Güncellenmiş)

Bu modül için, açık kaynaklı ve güçlü NLP kütüphanelerini kullanmaya devam edeceğiz.

#### Temel Metin Özellikleri Çıkarımı
**Teknoloji:** SpaCy, NLTK, Python standart kütüphaneleri (string işlemleri, regex).

**Rolü:** Makale metninin temizlenmesi, tokenizasyon, lemmatizasyon gibi temel ön işleme adımlarının yapılması ve metin uzunluğu, kelime sayısı gibi basit sayısal özelliklerin çıkarılması.

**Neden:** Metin tabanlı özelliklerin çıkarılması için temel ve güçlü araçlardır.

#### Gelişmiş Metin Nitelikleri Analizi (Duygu, Ton, Karmaşıklık)
**Teknoloji:** Hugging Face Transformers Kütüphanesi (özellikle duygu analizi ve ton analizi için önceden eğitilmiş modeller, örn. distilbert-base-uncased-finetuned-sst-2-english veya cardiffnlp/twitter-roberta-base-sentiment), TextStat (metin okunabilirliği için).

**Rolü:**
- **Duygu Analizi:** Makale metninin genel duygu tonunu belirlemek (pozitif, negatif, nötr).
- **Ton Analizi / Subjektiflik Tespiti:** Metnin ne kadar objektif veya sübjektif olduğunu analiz etmek. (Basit bir yaklaşım, objektif kelime listelerine dayalı skorlama veya BERT tabanlı modellerle sübjektiflik/objektiflik sınıflandırması olabilir.)
- **Dil Karmaşıklığı ve Okunabilirlik:** Metnin anlaşılma zorluğunu ve dilbilimsel karmaşıklık düzeyini değerlendirmek.

**Neden:** Transformer tabanlı modeller, metnin ince nüanslarını yakalayarak daha doğru duygu ve ton analizi yapabilir. TextStat, çeşitli okunabilirlik skorlarını hesaplamak için ücretsiz ve kolay kullanılabilir bir kütüphanedir.

#### Veri Yapılandırma
**Teknoloji:** Python (sözlükler, listeler).

**Rolü:** Analizlerden elde edilen tüm nitelikleri, makalenin diğer meta verileriyle birlikte yapılandırılmış bir formatta (JSON gibi) birleştirmek.

**Neden:** Modül çıktısının diğer modüllerle uyumlu ve kolayca işlenebilir olmasını sağlar.

### 4.2 Analiz Kriterleri (Çıkarılacak Nitelikler - Attributes)

Makale metninden çıkarılacak temel nitelikler:

#### 1. Metin Duygusu (Sentiment)
**Tanım:** Makale metninin genel olarak pozitif, negatif veya nötr bir duygu tonu taşıyıp taşımadığının belirlenmesi.

**Nasıl Elde Edilir:** Hugging Face Transformers'tan duygu analizi için eğitilmiş modeller (örn. distilbert-base-uncased-finetuned-sst-2-english veya cardiffnlp/twitter-roberta-base-sentiment). Model, metne bir duygu etiketi ve güven puanı atayacaktır.

#### 2. Metin Tarafsızlığı/Subjektifliği (Objectivity/Subjectivity)
**Tanım:** Metnin ne kadar tarafsız (gerçeklere dayalı) veya sübjektif (yazarın görüşlerini içeren) olduğunun bir göstergesi.

**Nasıl Elde Edilir:**
- **Transformer Tabanlı Sınıflandırma:** Duygu analizi modellerine benzer şekilde, sübjektiflik/objektiflik üzerine eğitilmiş özel bir Transformer modeli kullanılabilir (eğer mevcutsa).
- **Kural Tabanlı/Lexicon Yaklaşımı:** Önceden tanımlanmış sübjektif kelime (örn. "bence", "inanıyorum ki", "şok edici") ve objektif kelime listeleri kullanarak metindeki kullanım yoğunluğuna göre bir skor hesaplanabilir. (NLTK'nin Opinion Lexicon'u gibi kaynaklar kullanılabilir.)

#### 3. Dilin Karmaşıklığı ve Okunabilirlik
**Tanım:** Metnin ortalama bir okuyucu tarafından ne kadar kolay anlaşılabileceğinin ölçüsü. Genellikle eğitim seviyesi veya not düzeyi ile ilişkilendirilir.

**Nasıl Elde Edilir:** TextStat kütüphanesi kullanılarak Flesch-Kincaid Okunabilirlik Testi, Dale-Chall Okunabilirlik Formülü, Gunning Fog İndeksi gibi standart skorlar hesaplanır. Yüksek skorlar daha karmaşık bir dil anlamına gelir.

#### 4. Metin Uzunluğu ve Yoğunluğu
**Tanım:** Makalenin kelime sayısı ve cümle sayısı gibi temel metrikler. Prompt'tan gelen anahtar kelimelerin makale metni içindeki yoğunluğu ve dağılımı (Modül 3'ten alınan alaka düzeyi skoruyla birlikte değerlendirilebilir).

**Nasıl Elde Edilir:** SpaCy veya Python'un string metotları kullanılarak basit sayımlar.

### 4.3 Algoritma Akışı (Sistematik ve Akıllıca)

İşte Modül 3'ten gelen her bir alakalı makale için uygulanacak metin nitelikleri analiz süreci:

#### 1. Makale Verilerinin Alınması
Modül 3'ten, temizlenmiş makale metni ve ilgili meta verileri (URL, başlık, yazar, yayın tarihi, alaka düzeyi skoru) alınır.

#### 2. Gelişmiş Metin Niteliklerinin Çıkarılması
Alınan makale metni üzerinde aşağıdaki analizler yapılır:

- **Duygu Analizi:** Hugging Face Transformer duygu analizi modeli kullanılarak metnin duygusal tonu (pozitif, negatif, nötr) ve ilgili skorlar belirlenir.
- **Tarafsızlık/Subjektiflik Analizi:** Metnin ne kadar sübjektif veya objektif olduğu belirlenir. Bu, ayrı bir Transformer modeli ile veya kural tabanlı bir yaklaşımla yapılabilir.
- **Okunabilirlik Skorları:** TextStat kütüphanesi kullanılarak metnin Flesch-Kincaid, Dale-Chall vb. okunabilirlik skorları hesaplanır.
- **Temel Metin Metrikleri:** Metin uzunluğu (kelime ve karakter sayısı), cümle sayısı, ortalama kelime uzunluğu gibi temel istatistikler çıkarılır.

#### 3. Çıkarılan Niteliklerin Yapılandırılması
Tüm bu analizlerden elde edilen duygu skoru, tarafsızlık/sübjektiflik skoru, okunabilirlik skorları ve temel metin metrikleri, makalenin mevcut meta verilerine (URL, başlık, özet, yayın tarihi, alaka düzeyi skoru) eklenir.

#### 4. Nihai Makale Verilerinin Hazırlanması
Bu modülün çıktısı, kullanıcının prompt'una hem anlamsal olarak alakalı olduğu onaylanmış hem de detaylı metin nitelikleri (duygu, tarafsızlık, okunabilirlik, vb.) ile zenginleştirilmiş, temizlenmiş ve yapılandırılmış makale içeriklerinin ve meta verilerinin son listesi olacaktır. Bu liste, API'nin nihai çıktı olarak sunulmaya hazırdır.

Bu sadeleştirme ve odaklanma, projenin ilk sürümünü daha yönetilebilir hale getirecek ve temel "akıllı bilgi toplama" yeteneğini daha hızlı sunmanıza olanak tanıyacaktır. Daha sonra, projenin sonraki aşamalarında güvenilirlik değerlendirmesini ayrı ve kapsamlı bir modül olarak ele alabilirsiniz.



## Kurulum

### Gereksinimler
- Python 3.8+
- FastAPI
- Uvicorn
- SpaCy
- KeyBERT
- Transformers
- Scikit-learn
- NumPy

### Kurulum Adımları
1. Projeyi klonlayın
2. Sanal ortam oluşturun: `python -m venv venv`
3. Sanal ortamı aktifleştirin: `source venv/bin/activate` (Linux/Mac) veya `venv\Scripts\activate` (Windows)
4. Bağımlılıkları yükleyin: `pip install -r requirements.txt`
5. SpaCy modelini indirin: `python -m spacy download en_core_web_sm`
6. Uygulamayı çalıştırın: `uvicorn app.main:app --reload`

## API Endpoints

### Modül 1: Anahtar Kelime Çıkarımı
- `POST /api/v1/keywords/extract` - Anahtar kelime çıkarımı
- `POST /api/v1/keywords/expand` - Anlamsal genişletme
- `GET /api/v1/keywords/status` - İşlem durumu



## Kullanım Örnekleri

### Tam Araştırma Zinciri (Modül 1 → 2 → 3 → 4 → 5)

#### Modül 4: Detaylı Analiz
```json
POST /api/v1/text-quality/analyze-from-prompt
{
    "prompt": "Yapay zeka etiği ve şeffaflık konularındaki son gelişmeler",
    "max_keywords": 15,
    "max_urls": 20,
    "time_limit_days": 30,
    "relevance_threshold": 0.1,
    "max_content_length": 2048,
    "excluded_domains": [],
    "required_keywords": [],
    "language": "en"
}
```

#### Modül 5: Kapsamlı Raporlama
```json
POST /api/v1/query
{
    "prompt": "Yapay zeka etiği ve şeffaflık konularındaki son gelişmeler",
    "max_results": 5,
    "time_limit_days": 30,
    "max_keywords": 15,
    "max_urls": 20,
    "relevance_threshold": 0.1,
    "max_content_length": 2048,
    "generate_markdown": true,
    "language": "en"
}
```

### Tek Metin Analizi (Modül 4)
```json
POST /api/v1/text-quality/analyze-single-text
{
    "text": "Artificial intelligence has the potential to transform industries..."
}
```

### Anahtar Kelime Çıkarımı
```json
POST /module1/extract-keywords
{
    "prompt": "Yapay zeka etiği ve şeffaflık konularındaki son gelişmeler",
    "max_keywords": 20,
    "use_pos_filtering": true,
    "use_ner_filtering": true,
    "use_semantic_expansion": true
}
```

### Kaynak Arama
```json
POST /module2/search-from-prompt
{
    "prompt": "Yapay zeka etiği",
    "keywords": ["artificial intelligence", "ethics", "transparency"],
    "max_results": 50,
    "time_filter": "past_month"
}
```

### Modül 2: Dinamik Kaynak Keşfi
- `POST /api/v1/sources/search` - Kaynak arama
- `POST /api/v1/sources/crawl` - Web tarama
- `POST /api/v1/sources/search-and-crawl` - Arama ve tarama
- `GET /api/v1/sources/status` - İşlem durumu

### Modül 3: İçerik Analizi ve Alaka Düzeyi Kontrolü
- `POST /api/v1/content-analysis/analyze-from-prompt` - **Ana Endpoint**: Modül 1 → 2 → 3 zincirleme araştırma
- `GET /api/v1/content-analysis/health` - Sağlık kontrolü

### Modül 4: Gelişmiş Metin Nitelikleri Analizi
- `POST /api/v1/text-quality/analyze-from-prompt` - **Ana Endpoint**: Modül 1 → 2 → 3 → 4 tam zincirleme analiz
- `GET /api/v1/text-quality/health` - Sağlık kontrolü
- `POST /api/v1/text-quality/analyze-single-text` - Tek metin analizi
- `GET /api/v1/text-quality/stats` - Modül istatistikleri

## Geliştirme

### Modül Ekleme
1. `app/modules/` altında yeni modül klasörü oluşturun
2. `models.py`, `schemas.py`, `services.py`, `routes.py` dosyalarını ekleyin
3. Ana router'a modülü kaydedin

### Test
```bash
pytest tests/
```

## Lisans
MIT 
