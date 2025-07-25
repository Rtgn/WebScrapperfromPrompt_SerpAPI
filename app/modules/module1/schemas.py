from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class KeywordExtractionRequest(BaseModel):
    """Request schema for keyword extraction"""
    prompt: str = Field(..., description="Input text for keyword extraction", min_length=10)
    max_keywords: Optional[int] = Field(20, description="Maximum number of keywords to extract", ge=1, le=100)
    use_pos_filtering: Optional[bool] = Field(True, description="Use POS tagging for filtering")
    use_ner_filtering: Optional[bool] = Field(True, description="Use Named Entity Recognition for filtering")
    use_semantic_expansion: Optional[bool] = Field(True, description="Use semantic expansion for related terms")
    similarity_threshold: Optional[float] = Field(0.7, description="Similarity threshold for semantic expansion", ge=0.0, le=1.0)
    expansion_threshold: Optional[float] = Field(0.8, description="Threshold for semantic expansion", ge=0.0, le=1.0)

class KeywordInfo(BaseModel):
    """Individual keyword information"""
    keyword: str = Field(..., description="Extracted keyword")
    score: float = Field(..., description="Confidence score for the keyword", ge=0.0, le=1.0)
    source: str = Field(..., description="Source of extraction (spacy, keybert, ner, expansion)")
    pos_tag: Optional[str] = Field(None, description="Part of speech tag")
    entity_type: Optional[str] = Field(None, description="Named entity type if applicable")
    frequency: Optional[int] = Field(None, description="Frequency in the text")

class KeywordExtractionResponse(BaseModel):
    """Response schema for keyword extraction"""
    original_prompt: str = Field(..., description="Original input prompt")
    extracted_keywords: List[KeywordInfo] = Field(..., description="List of extracted keywords")
    total_keywords: int = Field(..., description="Total number of keywords extracted")
    processing_time: float = Field(..., description="Processing time in seconds")
    extraction_stats: Dict[str, Any] = Field(..., description="Statistics about the extraction process")
    semantic_clusters: Optional[List[List[str]]] = Field(None, description="Semantically related keyword clusters")

class KeywordExtractionError(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Detailed error information")
    error_code: str = Field(..., description="Error code for identification") 