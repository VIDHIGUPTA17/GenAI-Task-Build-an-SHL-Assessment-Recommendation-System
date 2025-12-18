
import torch
torch.classes.__path__ = []  # REQUIRED for SentenceTransformers + Streamlit

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Phase A imports
import sys
sys.path.append('.')
from embeddings.building_embedding import FAISSVectorDB, EmbeddingGenerator


# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIG
# ============================================================================
class Config:
    FAISS_INDEX_PATH = "embeddings_output/assessments.faiss"
    METADATA_PATH = "embeddings_output/metadata.pkl"
    CLEANED_JSON_PATH = "embeddings_output/cleaned_assessments.json"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # API Response limits
    MIN_RECOMMENDATIONS = 1
    MAX_RECOMMENDATIONS = 10
    DEFAULT_K = 10


# ============================================================================
# DATA MODELS
# ============================================================================
@dataclass
class RecommendedAssessment:
    """Single recommended assessment"""
    url: str
    adaptive_support: str  # "Yes"/"No"
    description: str
    duration: int  # in minutes
    remote_support: str  # "Yes"/"No"
    test_type: List[str]  # Array of test types
    
    def to_dict(self):
        return asdict(self)


@dataclass
class RecommendationResponse:
    """API response structure"""
    recommended_assessments: List[RecommendedAssessment]
    
    def to_dict(self):
        return {
            "recommended_assessments": [a.to_dict() for a in self.recommended_assessments]
        }


# ============================================================================
# ASSESSMENT METADATA EXTRACTOR
# ============================================================================
class MetadataExtractor:
    """Extract and normalize assessment metadata"""
    
    @staticmethod
    def extract_duration(metadata: Dict) -> int:
        import re
    
        duration_candidates = [
        metadata.get("duration"),
        metadata.get("Time (Minutes)"),   
        metadata.get("time_minutes"),
        metadata.get("Duration"),
    ]
    
        for duration_str in duration_candidates:
        # ✅ Skip null values
            if duration_str is None or duration_str == "":
                continue
        
        # ✅ Handle integers
            if isinstance(duration_str, int) and duration_str > 0:
                return duration_str
        
        # ✅ Handle strings (THE KEY FIX!)
            if isinstance(duration_str, str) and duration_str.strip():
                match = re.search(r'(\d+)', str(duration_str))
                if match:
                    value = int(match.group(1))
                    if value > 0:
                        return value
    
        return 0

    
    @staticmethod
    def extract_adaptive_support(metadata: Dict) -> str:
        """Extract adaptive support information"""
        adaptive = metadata.get("adaptive_support", "No")
        return "Yes" if adaptive and str(adaptive).lower() in ["yes", "true", "1"] else "No"
    
    @staticmethod
    def extract_remote_support(metadata: Dict) -> str:
        """Extract remote support information"""
        remote = metadata.get("remote_support", "No")
        return "Yes" if remote and str(remote).lower() in ["yes", "true", "1"] else "No"
    
    @staticmethod
    def extract_test_type(metadata: Dict) -> List[str]:
        """Extract test type as list"""
        test_type = metadata.get("test_type", "")
        
        if isinstance(test_type, list):
            return test_type
        
        if isinstance(test_type, str):
            # Split by comma if multiple, return as list
            types = [t.strip() for t in test_type.split(",") if t.strip()]
            return types if types else ["Other"]
        
        return ["Other"]
    
    @staticmethod
    def get_description(metadata: Dict) -> str:
        """Get assessment description"""
        return metadata.get("description", "Assessment for skill evaluation")


# ============================================================================
# QUERY PARSER - Extract constraints from query
# ============================================================================
class QueryParser:
    """Parse duration and other constraints from query"""
    
    @staticmethod
    def extract_duration_constraint(query: str) -> Optional[int]:
        """Extract max duration from query"""
        import re
        
        # Pattern: "completed in 40 minutes", "budget is about an hour", etc.
        patterns = [
            r'(\d+)\s*(?:minutes|mins|min)',
            r'about\s+an?\s+(\d*\.?\d+)?\s*hour',  # "about an hour" or "about 1.5 hours"
            r'budget.*?(\d+)\s*(?:minutes|mins|min)',
            r'duration.*?(?:at most|up to|max|maximum)?\s*(\d+)\s*(?:minutes|mins|min)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                value = match.group(1)
                if value:
                    if "hour" in query.lower() and not "minute" in query.lower():
                        # "about an hour" = 60 minutes
                        return 60
                    return int(float(value))
        
        return None
    
    @staticmethod
    def extract_skills(query: str) -> List[str]:
        """Extract technical skills from query"""
        import re
        
        skills_keywords = {
            "java": r"java(?:\s+8|advanced)?",
            "python": r"python",
            "sql": r"sql",
            "javascript": r"javascript|js",
            "c#": r"c#|csharp",
            "selenium": r"selenium",
            "testing": r"test(?:ing)?",
            "leadership": r"leadership|leader",
            "communication": r"communicat(?:ion|e)",
            "data": r"data",
            "sales": r"sales",
            "marketing": r"marketing",
        }
        
        found_skills = []
        for skill, pattern in skills_keywords.items():
            if re.search(pattern, query, re.IGNORECASE):
                found_skills.append(skill)
        
        return found_skills


# ============================================================================
# RECOMMENDATION ENGINE
# ============================================================================
class RecommendationEngine:
    """Main recommendation engine"""
    
    def __init__(self):
        logger.info("Loading Recommendation Engine...")
        
        # Load FAISS index
        self.vector_db = FAISSVectorDB.load(
            Config.FAISS_INDEX_PATH,
            Config.METADATA_PATH
        )
        logger.info(f"✓ Loaded FAISS index with {self.vector_db.index.ntotal} assessments")
        
        # Load embedding generator
        self.embedder = EmbeddingGenerator(
            model_name=Config.EMBEDDING_MODEL
        )
        logger.info(f"✓ Loaded embedding model: {Config.EMBEDDING_MODEL}")
        
        # Load cleaned JSON
        with open(Config.CLEANED_JSON_PATH, "r", encoding="utf-8") as f:
            self.assessments_json = json.load(f)
        logger.info(f"✓ Loaded {len(self.assessments_json)} assessments from JSON")
        
        # Create metadata lookup
        self.metadata_lookup = {m['chunk_id']: m for m in self.vector_db.metadata}
        
        logger.info("✓ Recommendation Engine Ready")
    
    def recommend(self, query: str, k: int = Config.DEFAULT_K) -> List[RecommendedAssessment]:
        """Generate recommendations"""
        
        logger.info(f"Query: {query}")
        logger.info(f"Requesting top {k} recommendations")
        
        # Parse query for constraints
        max_duration = QueryParser.extract_duration_constraint(query)
        logger.info(f"Duration constraint: {max_duration} minutes")
        
        # Encode query
        query_embedding = self.embedder.encode_single(query)
        
        # Search FAISS (get more results to filter by duration)
        search_k = min(k * 2, 100)  # Get 2x to filter by duration
        distances, metadatas = self.vector_db.search(query_embedding, k=search_k)
        
        results = []
        
        for distance, metadata in zip(distances, metadatas):
            # Extract full assessment info
            assessment_name = metadata.get("assessment_name", "Unknown")
            assessment_url = metadata.get("assessment_url", "")
            description = MetadataExtractor.get_description(metadata)
            duration = MetadataExtractor.extract_duration(metadata)
            adaptive_support = MetadataExtractor.extract_adaptive_support(metadata)
            remote_support = MetadataExtractor.extract_remote_support(metadata)
            test_type = MetadataExtractor.extract_test_type(metadata)
            
            # Apply duration filter if specified
            if max_duration and duration > max_duration:
                logger.info(f"Filtering out {assessment_name} (duration {duration}m > {max_duration}m)")
                continue
            
            # Create recommendation
            rec = RecommendedAssessment(
                url=assessment_url,
                adaptive_support=adaptive_support,
                description=description,
                duration=duration,
                remote_support=remote_support,
                test_type=test_type
            )
            
            results.append(rec)
            
            if len(results) >= k:
                break
        
        # Ensure minimum recommendations
        if not results:
            logger.warning("No results found, returning top recommendations without filters")
            distances, metadatas = self.vector_db.search(query_embedding, k=k)
            
            for distance, metadata in zip(distances, metadatas):
                rec = RecommendedAssessment(
                    url=metadata.get("assessment_url", ""),
                    adaptive_support=MetadataExtractor.extract_adaptive_support(metadata),
                    description=MetadataExtractor.get_description(metadata),
                    duration=MetadataExtractor.extract_duration(metadata),
                    remote_support=MetadataExtractor.extract_remote_support(metadata),
                    test_type=MetadataExtractor.extract_test_type(metadata)
                )
                results.append(rec)
        
        logger.info(f"✓ Generated {len(results)} recommendations")
        return results[:k]


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================
def create_api():
    """Create FastAPI application"""
    
    app = FastAPI(
        title="SHL Assessment Recommendation API",
        version="2.0",
        description="Intelligent assessment recommendation system"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize engine
    try:
        engine = RecommendationEngine()
    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}")
        raise
    
    # ========================================================================
    # ENDPOINTS
    # ========================================================================
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
        }
    
    @app.post("/recommend")
    async def recommend(
        query: str = Query(..., min_length=3, description="Job description or natural language query"),
        k: int = Query(Config.DEFAULT_K, ge=Config.MIN_RECOMMENDATIONS, le=Config.MAX_RECOMMENDATIONS, 
                      description="Number of recommendations (1-10)")
    ):
        """
        Get assessment recommendations for a query
        
        Request:
            - query: Job description or natural language query
            - k: Number of recommendations (1-10)
        
        Response:
            - recommended_assessments: List of recommended assessments with:
              - url: Assessment URL
              - adaptive_support: "Yes"/"No"
              - description: Assessment description
              - duration: Duration in minutes
              - remote_support: "Yes"/"No"
              - test_type: List of test types
        """
        
        try:
            if not query or len(query.strip()) < 3:
                raise HTTPException(
                    status_code=400,
                    detail="Query must be at least 3 characters"
                )
            
            # Generate recommendations
            results = engine.recommend(query, k=k)
            
            if not results:
                logger.warning(f"No recommendations found for query: {query}")
                return {
                    "recommended_assessments": []
                }
            
            # Format response
            response = {
                "recommended_assessments": [r.to_dict() for r in results]
            }
            
            logger.info(f"✓ Returned {len(results)} recommendations")
            return response
        
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )
    
    @app.post("/test-search")
    async def test_search():
        """Test endpoint with sample query"""
        sample_query = "Java developer with communication skills, max 40 minutes"
        results = engine.recommend(sample_query, k=5)
        return {
            "sample_query": sample_query,
            "count": len(results),
            "recommended_assessments": [r.to_dict() for r in results]
        }
    
    return app


# ============================================================================
# ENTRY POINT
# ============================================================================
# if __name__ == "__main__":
#     import sys
    
#     app = create_api()
    
#     logger.info(f"Starting API server on {Config.API_HOST}:{Config.API_PORT}")
    
#     uvicorn.run(
#         app,
#         host=Config.API_HOST,
#         port=Config.API_PORT,
#         log_level="info"
#     )