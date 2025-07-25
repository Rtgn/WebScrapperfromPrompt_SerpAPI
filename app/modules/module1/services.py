import spacy
import nltk
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import time
import re
from collections import Counter
import logging

from app.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedKeywordExtractor:
    def __init__(self):
        self.initialized = False
        self._initialize_models()
    
    def _initialize_models(self):
        try:
            logger.info("Initializing NLP models...")
            
            # Load SpaCy model
            logger.info(f"Loading SpaCy model: {settings.spacy_model}")
            self.nlp = spacy.load(settings.spacy_model)
            
            # Download NLTK data if not available
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
                nltk.download('stopwords')
                nltk.download('averaged_perceptron_tagger')
            
            # Initialize KeyBERT with SentenceTransformer
            logger.info(f"Loading KeyBERT model: {settings.keybert_model}")
            self.sentence_model = SentenceTransformer(settings.keybert_model)
            self.keybert = KeyBERT(model=self.sentence_model)
            
            # Initialize TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.95
            )
            
            try:
                self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            except Exception as e:
                logger.warning(f"Could not load sentiment analyzer: {e}")
                self.sentiment_analyzer = None
            
            self.initialized = True
            logger.info("All NLP models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def extract_keywords(self, prompt: str, max_keywords: int = 20, 
                        use_pos_filtering: bool = True, use_ner_filtering: bool = True,
                        use_semantic_expansion: bool = True, similarity_threshold: float = 0.7,
                        expansion_threshold: float = 0.8) -> Dict[str, Any]:
        start_time = time.time()
        
        if not self.initialized:
            raise RuntimeError("Keyword extractor not initialized")
        
        # Step 1: Text preprocessing with SpaCy
        logger.info("Step 1: Text preprocessing with SpaCy")
        doc = self.nlp(prompt.lower())
        
        # Step 2: Extract keywords using different methods
        keywords_data = {}
        
        # 2.1 SpaCy-based extraction (POS filtering + NER)
        if use_pos_filtering or use_ner_filtering:
            spacy_keywords = self._extract_spacy_keywords(doc, use_pos_filtering, use_ner_filtering)
            keywords_data['spacy'] = spacy_keywords
            logger.info(f"SpaCy extracted {len(spacy_keywords)} keywords")
        
        # 2.2 KeyBERT extraction
        keybert_keywords = self._extract_keybert_keywords(prompt, max_keywords)
        keywords_data['keybert'] = keybert_keywords
        logger.info(f"KeyBERT extracted {len(keybert_keywords)} keywords")
        
        # 2.3 TF-IDF extraction
        tfidf_keywords = self._extract_tfidf_keywords(prompt, max_keywords)
        keywords_data['tfidf'] = tfidf_keywords
        logger.info(f"TF-IDF extracted {len(tfidf_keywords)} keywords")
        
        # Step 3: Combine and rank keywords
        logger.info("Step 3: Combining and ranking keywords")
        combined_keywords = self._combine_keywords(keywords_data, max_keywords)
        
        # Step 4: Semantic expansion (optional)
        expanded_keywords = []
        if use_semantic_expansion and combined_keywords:
            logger.info("Step 4: Semantic expansion")
            expanded_keywords = self._semantic_expansion(
                combined_keywords, prompt, expansion_threshold
            )
            combined_keywords.extend(expanded_keywords)
        
        # Step 5: Final ranking and deduplication
        logger.info("Step 5: Final ranking and deduplication")
        final_keywords = self._final_ranking_and_deduplication(combined_keywords, max_keywords)
        
        # Final safety check: ensure all scores are within 0-1 range
        for kw in final_keywords:
            kw['score'] = max(0.0, min(1.0, kw['score']))
        
        # Step 6: Create semantic clusters
        semantic_clusters = self._create_semantic_clusters(final_keywords, similarity_threshold)
        
        processing_time = time.time() - start_time
        
        # Prepare response
        response = {
            'original_prompt': prompt,
            'extracted_keywords': final_keywords,
            'total_keywords': len(final_keywords),
            'processing_time': processing_time,
            'extraction_stats': {
                'spacy_keywords': len(keywords_data.get('spacy', [])),
                'keybert_keywords': len(keywords_data.get('keybert', [])),
                'tfidf_keywords': len(keywords_data.get('tfidf', [])),
                'expanded_keywords': len(expanded_keywords) if use_semantic_expansion else 0,
                'semantic_clusters': len(semantic_clusters)
            },
            'semantic_clusters': semantic_clusters
        }
        
        logger.info(f"Keyword extraction completed in {processing_time:.2f} seconds")
        return response
    
    def _extract_spacy_keywords(self, doc, use_pos_filtering: bool, use_ner_filtering: bool) -> List[Dict[str, Any]]:
        keywords = []
        
        # POS-based extraction
        if use_pos_filtering:
            for token in doc:
                if (token.pos_ in settings.allowed_pos_tags and 
                    len(token.text) >= settings.min_keyword_length and
                    not token.is_stop and not token.is_punct):
                    
                    # Lemmatize the token
                    lemma = token.lemma_.lower()
                    
                    # Check if it's a meaningful word
                    if self._is_meaningful_word(lemma):
                        keywords.append({
                            'keyword': lemma,
                            'score': 0.8,
                            'source': 'spacy',
                            'pos_tag': token.pos_,
                            'entity_type': None,
                            'frequency': 1
                        })
            
            # Also extract noun phrases
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.lower().strip()
                if len(chunk_text) >= settings.min_keyword_length and self._is_meaningful_word(chunk_text):
                    keywords.append({
                        'keyword': chunk_text,
                        'score': 0.9,
                        'source': 'spacy_noun_chunks',
                        'pos_tag': 'NOUN_PHRASE',
                        'entity_type': None,
                        'frequency': 1
                    })
        
        # NER-based extraction
        if use_ner_filtering:
            for ent in doc.ents:
                if ent.label_ in settings.ner_entity_types:
                    # Clean entity text
                    entity_text = re.sub(r'[^\w\s]', '', ent.text).strip().lower()
                    if len(entity_text) >= settings.min_keyword_length:
                        keywords.append({
                            'keyword': entity_text,
                            'score': 0.9,
                            'source': 'ner',
                            'pos_tag': None,
                            'entity_type': ent.label_,
                            'frequency': 1
                        })
        
        return keywords
    
    def _extract_keybert_keywords(self, text: str, max_keywords: int) -> List[Dict[str, Any]]:
        try:
            # Clean and prepare text for KeyBERT
            cleaned_text = text.strip()
            if len(cleaned_text) < 10:
                logger.warning("Text too short for KeyBERT extraction")
                return []
            
            keywords = []
            
            # Method 1: Basic extraction
            try:
                keywords_with_scores = self.keybert.extract_keywords(
                    cleaned_text, 
                    keyphrase_ngram_range=(1, 4),
                    stop_words='english',
                    use_maxsum=True,
                    nr_candidates=max_keywords * 3,
                    top_n=max_keywords * 2,
                    diversity=0.6
                )
                
                for keyword, score in keywords_with_scores:
                    if len(keyword) >= settings.min_keyword_length:
                        keywords.append({
                            'keyword': keyword.lower(),
                            'score': float(score),
                            'source': 'keybert',
                            'pos_tag': None,
                            'entity_type': None,
                            'frequency': 1
                        })
            except Exception as e:
                logger.warning(f"KeyBERT method 1 failed: {e}")
            
            if not keywords:
                try:
                    keywords_with_scores = self.keybert.extract_keywords(
                        cleaned_text,
                        keyphrase_ngram_range=(1, 2),
                        stop_words='english',
                        use_maxsum=False,
                        nr_candidates=max_keywords,
                        top_n=max_keywords
                    )
                    
                    for keyword, score in keywords_with_scores:
                        if len(keyword) >= settings.min_keyword_length:
                            keywords.append({
                                'keyword': keyword.lower(),
                                'score': float(score),
                                'source': 'keybert_alt',
                                'pos_tag': None,
                                'entity_type': None,
                                'frequency': 1
                            })
                except Exception as e:
                    logger.warning(f"KeyBERT method 2 failed: {e}")
            
            # Method 3: Simple extraction as fallback
            if not keywords:
                try:
                    # Use sentence embeddings to find important words
                    doc = self.nlp(cleaned_text)
                    important_words = []
                    
                    for token in doc:
                        if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                            len(token.text) >= settings.min_keyword_length and
                            not token.is_stop and not token.is_punct):
                            important_words.append(token.lemma_.lower())
                    
                    # Get embeddings for important words
                    if important_words:
                        word_embeddings = self.sentence_model.encode(important_words)
                        text_embedding = self.sentence_model.encode([cleaned_text])
                        
                        # Calculate similarities
                        similarities = cosine_similarity(word_embeddings, text_embedding).flatten()
                        
                        # Create keyword-score pairs
                        word_scores = list(zip(important_words, similarities))
                        word_scores.sort(key=lambda x: x[1], reverse=True)
                        
                        for word, score in word_scores[:max_keywords]:
                            keywords.append({
                                'keyword': word,
                                'score': float(score),
                                'source': 'keybert_fallback',
                                'pos_tag': None,
                                'entity_type': None,
                                'frequency': 1
                            })
                except Exception as e:
                    logger.warning(f"KeyBERT fallback method failed: {e}")
            
            logger.info(f"KeyBERT extracted {len(keywords)} keywords")
            return keywords
            
        except Exception as e:
            logger.error(f"Error in KeyBERT extraction: {e}")
            return []
    
    def _extract_tfidf_keywords(self, text: str, max_keywords: int) -> List[Dict[str, Any]]:
        try:
            # Clean and prepare text
            cleaned_text = text.strip()
            if len(cleaned_text) < 10:
                logger.warning("Text too short for TF-IDF extraction")
                return []
            
            tfidf_vectorizer = TfidfVectorizer(
                max_features=200,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=1,
                max_df=1.0,
                lowercase=True,
                token_pattern=r'\b[a-zA-Z]{2,}\b'
            )
            
            tfidf_matrix = tfidf_vectorizer.fit_transform([cleaned_text])
            feature_names = tfidf_vectorizer.get_feature_names_out()
            
            # Get TF-IDF scores
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Create keyword-score pairs
            keyword_scores = list(zip(feature_names, tfidf_scores))
            
            # Sort by score and filter
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            keywords = []
            for keyword, score in keyword_scores[:max_keywords]:
                if len(keyword) >= settings.min_keyword_length and score > 0:
                    # Normalize score to 0-1 range
                    max_score = max(tfidf_scores) if tfidf_scores.max() > 0 else 1.0
                    normalized_score = min(score / max_score, 1.0)
                    
                    # Boost score for important terms
                    
                    
                    keywords.append({
                        'keyword': keyword.lower(),
                        'score': float(normalized_score),
                        'source': 'tfidf',
                        'pos_tag': None,
                        'entity_type': None,
                        'frequency': 1
                    })
            
            # If no keywords found, try with more lenient settings
            if not keywords:
                logger.info("TF-IDF no keywords found, trying with more lenient settings")
                tfidf_vectorizer_relaxed = TfidfVectorizer(
                    max_features=50,
                    stop_words='english',
                    ngram_range=(1, 1),
                    min_df=1,
                    max_df=1.0,
                    lowercase=True
                )
                
                tfidf_matrix_relaxed = tfidf_vectorizer_relaxed.fit_transform([cleaned_text])
                feature_names_relaxed = tfidf_vectorizer_relaxed.get_feature_names_out()
                tfidf_scores_relaxed = tfidf_matrix_relaxed.toarray()[0]
                
                keyword_scores_relaxed = list(zip(feature_names_relaxed, tfidf_scores_relaxed))
                keyword_scores_relaxed.sort(key=lambda x: x[1], reverse=True)
                
                for keyword, score in keyword_scores_relaxed[:max_keywords//2]:
                    if len(keyword) >= settings.min_keyword_length and score > 0:
                        max_score = max(tfidf_scores_relaxed) if tfidf_scores_relaxed.max() > 0 else 1.0
                        normalized_score = min(score / max_score, 1.0)
                        
                        keywords.append({
                            'keyword': keyword.lower(),
                            'score': float(normalized_score),
                            'source': 'tfidf_relaxed',
                            'pos_tag': None,
                            'entity_type': None,
                            'frequency': 1
                        })
            
            logger.info(f"TF-IDF extracted {len(keywords)} keywords")
            return keywords
            
        except Exception as e:
            logger.error(f"Error in TF-IDF extraction: {e}")
            return []
    
    def _combine_keywords(self, keywords_data: Dict[str, List], max_keywords: int) -> List[Dict[str, Any]]:
        all_keywords = []
        
        # Collect all keywords
        for source, keywords in keywords_data.items():
            all_keywords.extend(keywords)
        
        # Group by keyword and combine scores
        keyword_groups = {}
        for kw in all_keywords:
            keyword = kw['keyword']
            if keyword not in keyword_groups:
                keyword_groups[keyword] = {
                    'keyword': keyword,
                    'score': 0.0,
                    'sources': [],
                    'pos_tag': None,
                    'entity_type': None,
                    'frequency': 0
                }
            
            # Add score from this source
            keyword_groups[keyword]['score'] += kw['score']
            keyword_groups[keyword]['sources'].append(kw['source'])
            keyword_groups[keyword]['frequency'] += kw.get('frequency', 1)
            
            # Keep POS tag and entity type if available
            if kw.get('pos_tag'):
                keyword_groups[keyword]['pos_tag'] = kw['pos_tag']
            if kw.get('entity_type'):
                keyword_groups[keyword]['entity_type'] = kw['entity_type']
        
        # Normalize scores and create final list
        max_score = max(kw['score'] for kw in keyword_groups.values()) if keyword_groups else 1.0
        
        # Prevent division by zero
        if max_score <= 0:
            max_score = 1.0
        
        combined_keywords = []
        for keyword_data in keyword_groups.values():
            # Normalize score
            normalized_score = keyword_data['score'] / max_score
            
            # Boost score based on number of sources (but keep it reasonable)
            source_boost = min(len(keyword_data['sources']) * 0.05, 0.2)
            final_score = normalized_score + source_boost
            
            # Ensure score is within 0-1 range
            final_score = max(0.0, min(1.0, final_score))
            
            combined_keywords.append({
                'keyword': keyword_data['keyword'],
                'score': final_score,
                'source': '+'.join(keyword_data['sources']),
                'pos_tag': keyword_data['pos_tag'],
                'entity_type': keyword_data['entity_type'],
                'frequency': keyword_data['frequency']
            })
        
        # Sort by score and limit
        combined_keywords.sort(key=lambda x: x['score'], reverse=True)
        return combined_keywords[:max_keywords]
    
    def _semantic_expansion(self, keywords: List[Dict], original_text: str, threshold: float) -> List[Dict]:
        if not keywords:
            return []
        
        try:
            expanded_keywords = []
            
            # Get keyword embeddings
            keyword_texts = [kw['keyword'] for kw in keywords]
            keyword_embeddings = self.sentence_model.encode(keyword_texts)
            
            # Get original text embedding
            text_embedding = self.sentence_model.encode([original_text])
            
            # Calculate similarities
            similarities = cosine_similarity(keyword_embeddings, text_embedding).flatten()
            
            # Method 1: High similarity keywords
            for i, similarity in enumerate(similarities):
                if similarity >= threshold:
                    expanded_kw = keywords[i].copy()
                    expanded_kw['score'] = expanded_kw['score'] * 0.8
                    # Ensure score is within 0-1 range
                    expanded_kw['score'] = max(0.0, min(1.0, expanded_kw['score']))
                    expanded_kw['source'] = expanded_kw['source'] + '+expansion'
                    expanded_keywords.append(expanded_kw)
            
            # Method 2: Related keyword pairs
            if len(keywords) > 1:
                keyword_similarity_matrix = cosine_similarity(keyword_embeddings)
                related_pairs = []
                
                for i in range(len(keywords)):
                    for j in range(i + 1, len(keywords)):
                        if keyword_similarity_matrix[i][j] >= threshold * 0.8:
                            related_pairs.append((keywords[i], keywords[j], keyword_similarity_matrix[i][j]))
                
                for kw1, kw2, sim_score in related_pairs[:8]:
                    combined_term = f"{kw1['keyword']} {kw2['keyword']}"
                    if len(combined_term) >= settings.min_keyword_length:
                        score = (kw1['score'] + kw2['score']) * 0.6
                        score = max(0.0, min(1.0, score))
                        expanded_keywords.append({
                            'keyword': combined_term,
                            'score': score,
                            'source': 'semantic_combination',
                            'pos_tag': None,
                            'entity_type': None,
                            'frequency': 1
                        })
            
            # Method 3: Domain-specific expansion
            domain_keywords = self._get_domain_keywords(original_text)
            for domain_kw in domain_keywords:
                # Check if domain keyword is similar to any existing keyword
                domain_embedding = self.sentence_model.encode([domain_kw])
                domain_similarities = cosine_similarity(domain_embedding, keyword_embeddings).flatten()
                
                if max(domain_similarities) >= threshold * 0.7:
                    score = max(domain_similarities) * 0.7
                    score = max(0.0, min(1.0, score))
                    expanded_keywords.append({
                        'keyword': domain_kw,
                        'score': score,
                        'source': 'domain_expansion',
                        'pos_tag': None,
                        'entity_type': None,
                        'frequency': 1
                    })
            
            # Method 4: Synonym expansion
            synonym_keywords = self._get_synonyms(keywords)
            expanded_keywords.extend(synonym_keywords)
            
            logger.info(f"Semantic expansion added {len(expanded_keywords)} keywords")
            return expanded_keywords
            
        except Exception as e:
            logger.error(f"Error in semantic expansion: {e}")
            return []
    
    def _get_domain_keywords(self, text: str) -> List[str]:
        domain_keywords = []
        
        # AI and Ethics domain keywords
        ai_ethics_keywords = [
            "artificial intelligence", "machine learning", "deep learning", "neural networks",
            "algorithmic bias", "fairness", "accountability", "responsibility", "governance",
            "regulation", "privacy", "security", "trust", "explainability", "interpretability",
            "robustness", "reliability", "safety", "autonomy", "decision making",
            "transparency", "openness", "disclosure", "audit", "compliance", "standards",
            "guidelines", "principles", "values", "morality", "justice", "equity",
            "discrimination", "prejudice", "stereotyping", "inclusivity", "diversity"
        ]
        
        # Check which domain keywords are relevant
        text_lower = text.lower()
        for keyword in ai_ethics_keywords:
            if any(word in text_lower for word in keyword.split()):
                domain_keywords.append(keyword)
        
        return domain_keywords[:10]
    
    def _get_synonyms(self, keywords: List[Dict]) -> List[Dict]:
        """Get synonyms for existing keywords"""
        synonyms = []
        
        # Simple synonym mapping
        synonym_map = {
            "ai": ["artificial intelligence", "machine intelligence", "computational intelligence"],
            "ethics": ["morality", "principles", "values", "standards"],
            "transparency": ["openness", "clarity", "visibility", "disclosure"],
            "development": ["progress", "advancement", "evolution", "growth"],
            "recent": ["latest", "current", "modern", "contemporary"],
            "ethics": ["ethical", "moral", "principled"],
            "transparency": ["transparent", "open", "clear"]
        }
        
        for kw in keywords:
            keyword = kw['keyword']
            if keyword in synonym_map:
                for synonym in synonym_map[keyword]:
                    score = kw['score'] * 0.6  # Lower score for synonyms
                    # Ensure score is within 0-1 range
                    score = max(0.0, min(1.0, score))
                    synonyms.append({
                        'keyword': synonym,
                        'score': score,
                        'source': 'synonym_expansion',
                        'pos_tag': None,
                        'entity_type': None,
                        'frequency': 1
                    })
        
        return synonyms
    
    def _final_ranking_and_deduplication(self, keywords: List[Dict], max_keywords: int) -> List[Dict]:
        """Final ranking and deduplication of keywords"""
        # Remove exact duplicates
        seen_keywords = set()
        unique_keywords = []
        
        for kw in keywords:
            if kw['keyword'] not in seen_keywords:
                seen_keywords.add(kw['keyword'])
                unique_keywords.append(kw)
        
        # Sort by score and limit
        unique_keywords.sort(key=lambda x: x['score'], reverse=True)
        return unique_keywords[:max_keywords]
    
    def _create_semantic_clusters(self, keywords: List[Dict], similarity_threshold: float) -> List[List[str]]:
        if len(keywords) < 2:
            return []
        
        try:
            # Get keyword embeddings
            keyword_texts = [kw['keyword'] for kw in keywords]
            embeddings = self.sentence_model.encode(keyword_texts)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Create clusters
            clusters = []
            used_indices = set()
            
            for i in range(len(keywords)):
                if i in used_indices:
                    continue
                
                cluster = [keyword_texts[i]]
                used_indices.add(i)
                
                # Find similar keywords
                for j in range(i + 1, len(keywords)):
                    if j not in used_indices and similarity_matrix[i][j] >= similarity_threshold:
                        cluster.append(keyword_texts[j])
                        used_indices.add(j)
                
                if len(cluster) > 1:  # Only add clusters with multiple keywords
                    clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error creating semantic clusters: {e}")
            return []
    
    def _is_meaningful_word(self, word: str) -> bool:
        if len(word) < settings.min_keyword_length:
            return False
        
        # Check for common meaningless patterns
        meaningless_patterns = [
            r'^\d+$',  # Only numbers
            r'^[a-z]{1,2}$',  # Very short words
            r'^[^a-zA-Z]*$',  # No letters
        ]
        
        for pattern in meaningless_patterns:
            if re.match(pattern, word):
                return False
        
        return True

# Global instance
keyword_extractor = AdvancedKeywordExtractor() 