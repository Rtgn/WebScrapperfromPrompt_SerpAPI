# Advanced Research API

This project is a **modular FastAPI application** designed for:
- Advanced natural language processing (NLP)
- Keyword extraction
- Dynamic resource discovery

---

## **Project Structure**

### **Module 1: Advanced Keyword Extraction and Semantic Expansion**

#### **1.1 Technology Stack and Roles**

**SpaCy**  
- **Role:** Efficient preprocessing of the user prompt (tokenization, lemmatization, stopword filtering) and Named Entity Recognition (NER).  
- **Why:** Chosen for its speed, production-readiness, robust NER capabilities, and reliable NLP tools.

**KeyBERT**  
- **Role:** Extracting the most contextually and semantically relevant keywords/phrases from the preprocessed prompt text.  
- **Why:** Uses BERT-based models to understand semantic relationships, producing more relevant keywords and multi-word phrases (e.g., "new models", "breakthroughs").

**Hugging Face Transformers (Sentence-BERT)**  
- **Role:** Convert extracted keywords into semantic embeddings and expand them by finding similar concepts in a broader vocabulary.  
- **Why:** Sentence-BERT is optimized for measuring semantic similarity between phrases with high accuracy.

**Scikit-learn / NumPy**  
- **Role:** Compute cosine similarity between embeddings to assess semantic relevance.  
- **Why:** Industry-standard libraries for vector operations and similarity calculations.

---

#### **1.2 Algorithm Workflow (Systematic & Intelligent Steps)**

**Step 1: Receive Prompt Input**  
Accepts a free-text prompt via API.  
*Example:*  
`"Recent developments in AI, groundbreaking models, startup innovations, and discoveries."`

**Step 2: Preprocessing & Entity Recognition (with SpaCy)**  
- Load SpaCy model (`en_core_web_sm`)  
- Tokenize, lemmatize, and filter stopwords  
- Apply NER to extract named entities  
- Build **Core Keyword Pool** from identified entities.

**Step 3: Semantic Keyword Extraction (with KeyBERT)**  
- Use Sentence-BERT model like `all-MiniLM-L6-v2`  
- Extract top keywords and add to **Core Keyword Pool**.

**Step 4: Semantic Expansion (Transformers + Scikit-learn)**  
- Load Sentence-BERT model from Hugging Face  
- For each keyword, compute embeddings and find semantically similar concepts using cosine similarity.  
- Build **Extended Keyword Pool**.  
  *Examples:*  
  - `"artificial intelligence" → "machine learning", "deep learning", "neural networks"`  
  - `"startup innovation" → "fintech startups", "biotech", "entrepreneurial ecosystem"`

**Step 5: Final Query Generation**  
- Merge Core + Extended pools  
- Generate queries:  
  - Single terms: `"AI", "machine learning"`  
  - Phrases: `"new models"`  
  - Combinations: `"AI AND new models"`  
  - Long-tail queries: `"ethical challenges in startup AI systems"`

---

### **Module 2: Dynamic Source Discovery & Expanded Search**

#### **2.1 Technology Stack and Roles**

**Search Engine Results API**  
- **Technology:** Google Custom Search API or SerpApi  
- **Role:** Retrieve search result URLs and metadata.  
- **Why:** Avoids scraping difficulties, returns structured data.

**Web Crawler Framework**  
- **Technology:** Scrapy  
- **Role:** Crawl seed URLs and follow links.  
- **Why:** Large-scale, async crawling with middleware support.

**Dynamic Content Loading**  
- **Technology:** Scrapy-Splash or Selenium  
- **Role:** Execute JavaScript to retrieve full content.  
- **Why:** Needed for dynamic web pages.

**Proxy & User-Agent Management**  
- **Technology:** Proxy pools + Scrapy middlewares  
- **Role:** Prevent IP blocking and mimic human behavior.

**Temporary Data Storage**  
- **Technology:** Redis + PostgreSQL/MongoDB  
- **Role:** Manage URL queues and metadata storage.

---

#### **2.2 Workflow**

1. Receive search queries from **Module 1**.  
2. Use SERP API to get seed URLs (supports time filtering).  
3. Start Scrapy crawling with proxy/user-agent rotation.  
4. Filter irrelevant links (e.g., PDFs, outdated content).  
5. Prioritize based on:  
   - Keyword presence in URL/anchor text  
   - Semantic similarity (Sentence-BERT)  
   - Domain reputation  
   - Crawl depth  
6. Extract article metadata for the next module.

---

### **Module 3: Intelligent Content Analysis & Relevance Scoring**

#### **3.1 Technologies**
- **Content Parsing:** BeautifulSoup4, python-readability  
- **Semantic Relevance:** Hugging Face Transformers + PyTorch

#### **3.2 Workflow**
1. Parse and clean main text.  
2. Preprocess for Transformer input.  
3. Compute semantic similarity:  
   - Generate embeddings for both prompt & article  
   - Use cosine similarity → **Relevance Score**  
4. Filter low-relevance content.  
5. Store structured data: `title, text, summary, author, date, URL, score`.

---

### **Module 4: Advanced Text Attribute Analysis**

#### **4.1 Technologies**
- **Basic NLP:** SpaCy, NLTK, regex  
- **Sentiment & Subjectivity:** Transformers (e.g., `distilbert-sst-2`, `cardiffnlp/twitter-roberta-sentiment`)  
- **Readability:** TextStat  
- **Data Structuring:** Python dicts & JSON

#### **4.2 Extracted Attributes**
- **Sentiment:** Positive, negative, neutral tone  
- **Objectivity:** Subjective or factual text  
- **Readability:** Flesch-Kincaid, Dale-Chall, Gunning Fog scores  
- **Length & Density:** Word counts, keyword density

#### **4.3 Workflow**
1. Get articles from Module 3.  
2. Run sentiment and subjectivity models.  
3. Compute readability and text metrics.  
4. Merge results into enriched JSON with metadata.

---

## **Installation**

### **Requirements**
- Python 3.8+  
- FastAPI  
- Uvicorn  
- SpaCy  
- KeyBERT  
- Transformers  
- Scikit-learn  
- NumPy  

### **Steps**
```bash
# Clone repo
git clone <repo_url>

# Create virtual environment
python -m venv venv

# Activate venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Download SpaCy model
python -m spacy download en_core_web_sm

# Run the app
uvicorn app.main:app --reload

