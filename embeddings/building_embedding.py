

import json
import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Configuration for Phase A"""
    
    # Paths
    INPUT_JSON_PATH = "app/shl_catalog_enriched_20251218_120624.json"  # Your input JSON file
    OUTPUT_DIR = "embeddings_output"
    
    # Embedding Model
    # Using 'all-MiniLM-L6-v2' - lightweight, free, good performance
    # Other options: 'all-mpnet-base-v2' (larger, better), 'distiluse-base-multilingual-cased-v2' (multilingual)
    MODEL_NAME = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384  # Output dimension of all-MiniLM-L6-v2
    
    # Storage
    FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, "assessments.faiss")
    METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.pkl")
    CLEANED_JSON_PATH = os.path.join(OUTPUT_DIR, "cleaned_assessments.json")
    CHUNK_INFO_PATH = os.path.join(OUTPUT_DIR, "chunk_info.json")
    
    # Processing
    BATCH_SIZE = 32  # Batch size for embedding generation

# ============================================================================
# DATA CLEANING
# ============================================================================
class DataCleaner:
    """Clean and normalize assessment data"""
    
    @staticmethod
    def clean_string(text: str) -> str:
        """
        Clean a string by removing extra whitespace and trailing commas
        """
        if text is None:
            return ""
        
        text = str(text).strip()
        text = text.rstrip(',')  # Remove trailing commas
        text = ' '.join(text.split())  # Normalize whitespace
        return text
    
    @staticmethod
    def clean_assessment(assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean a single assessment record
        """
        cleaned = {}
        
        # Essential fields
        cleaned['Assessment Name'] = DataCleaner.clean_string(
            assessment.get('Assessment Name', '')
        )
        cleaned['Assessment URL'] = DataCleaner.clean_string(
            assessment.get('Assessment URL', '')
        )
        
        # Text fields
        cleaned['Description'] = DataCleaner.clean_string(
            assessment.get('Description', '')
        )
        cleaned['Test Type'] = DataCleaner.clean_string(
            assessment.get('Test Type', '')
        )
        cleaned['Job Levels'] = DataCleaner.clean_string(
            assessment.get('Job Levels', '')
        )
        cleaned['Languages'] = DataCleaner.clean_string(
            assessment.get('Languages', '')
        )
        
        # Boolean/Simple fields
        cleaned['Remote Testing Support'] = assessment.get('Remote Testing Support', 'No')
        cleaned['Adaptive/IRT Support'] = assessment.get('Adaptive/IRT Support', 'No')
        cleaned['Test Type Keys'] = assessment.get('Test Type Keys', [])
        cleaned['Time (Minutes)'] = assessment.get('Time (Minutes)', None)
        
        return cleaned
    
    @staticmethod
    def validate_assessment(assessment: Dict[str, Any]) -> bool:
        """
        Check if assessment has minimum required fields
        """
        required_fields = ['Assessment Name', 'Assessment URL', 'Description']
        for field in required_fields:
            if not assessment.get(field, '').strip():
                logger.warning(f"Missing {field} in assessment: {assessment.get('Assessment Name', 'Unknown')}")
                return False
        return True


# ============================================================================
# DATA CHUNKING & PREPARATION
# ============================================================================
class DataChunker:
    """
    Create text chunks from assessment data for embedding
    A chunk represents a single assessment with all its information combined
    """
    
    @staticmethod
    def create_chunk(assessment: Dict[str, Any]) -> str:
        """
        Create a single text chunk from an assessment
        
        Format:
        Title: [Name]
        Description: [Description]
        Type: [Test Type]
        Job Levels: [Levels]
        Languages: [Languages]
        Time: [Minutes]
        Remote Support: [Yes/No]
        """
        
        parts = []
        
        # Title (most important)
        if assessment.get('Assessment Name'):
            parts.append(f"Title: {assessment['Assessment Name']}")
        
        # Description (critical for understanding)
        if assessment.get('Description'):
            parts.append(f"Description: {assessment['Description']}")
        
        # Test Type
        if assessment.get('Test Type'):
            parts.append(f"Test Type: {assessment['Test Type']}")
        
        # Job Levels
        if assessment.get('Job Levels'):
            parts.append(f"Job Levels: {assessment['Job Levels']}")
        
        # Languages
        if assessment.get('Languages'):
            parts.append(f"Languages: {assessment['Languages']}")
        
        # Time
        if assessment.get('Time (Minutes)'):
            parts.append(f"Duration: {assessment['Time (Minutes)']} minutes")
        
        # Remote Support
        remote_support = assessment.get('Remote Testing Support', 'No')
        parts.append(f"Remote Support: {remote_support}")
        
        # Join all parts with newlines
        chunk = " | ".join(parts)
        return chunk
    
    @staticmethod
    def chunk_assessments(assessments: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Convert assessments into chunks and maintain metadata
        
        Returns:
            - chunks: List of text chunks (one per assessment)
            - metadata: List of metadata dicts (name, URL, original_assessment)
        """
        chunks = []
        metadata = []
        
        for idx, assessment in enumerate(assessments):
            chunk_text = DataChunker.create_chunk(assessment)
            chunks.append(chunk_text)
            
            metadata_entry = {
                'chunk_id': idx,
                'assessment_name': assessment['Assessment Name'],
                'assessment_url': assessment['Assessment URL'],
                'test_type': assessment['Test Type'],
                'job_levels': assessment['Job Levels'],
                'description': assessment['Description'],
                'time_minutes': assessment.get('Time (Minutes)'),
                'remote_support': assessment['Remote Testing Support'],
            }
            metadata.append(metadata_entry)
        
        logger.info(f"✓ Created {len(chunks)} chunks from {len(assessments)} assessments")
        return chunks, metadata


# ============================================================================
# EMBEDDING GENERATION
# ============================================================================
class EmbeddingGenerator:
    """Generate embeddings using HuggingFace models"""
    
    def __init__(self, model_name: str = Config.MODEL_NAME):
        """
        Initialize the embedding model
        
        Args:
            model_name: HuggingFace model identifier
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        logger.info(f"✓ Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def encode_batch(self, texts: List[str], batch_size: int = Config.BATCH_SIZE) -> np.ndarray:
        """
        Encode a batch of texts into embeddings
        
        Args:
            texts: List of text strings to encode
            batch_size: Number of texts to process at once
            
        Returns:
            np.ndarray: Array of shape (len(texts), embedding_dim)
        """
        logger.info(f"Encoding {len(texts)} texts with batch_size={batch_size}")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        logger.info(f"✓ Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text into an embedding
        
        Args:
            text: Text string to encode
            
        Returns:
            np.ndarray: Embedding vector of shape (embedding_dim,)
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding


# ============================================================================
# VECTOR DATABASE (FAISS)
# ============================================================================
class FAISSVectorDB:
    """Store and manage embeddings using FAISS"""
    
    def __init__(self, dimension: int = Config.EMBEDDING_DIM):
        """
        Initialize FAISS index
        
        Args:
            dimension: Embedding dimension
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
        self.metadata = []
        logger.info(f"✓ FAISS index initialized with dimension {dimension}")
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """
        Add embeddings to the index
        
        Args:
            embeddings: np.ndarray of shape (n_samples, dimension)
            metadata: List of metadata dicts corresponding to embeddings
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} != index dimension {self.dimension}")
        
        # Ensure embeddings are float32 (required by FAISS)
        embeddings = embeddings.astype(np.float32)
        
        self.index.add(embeddings)
        self.metadata.extend(metadata)
        
        logger.info(f"✓ Added {embeddings.shape[0]} embeddings to FAISS index (total: {self.index.ntotal})")
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[List[float], List[Dict[str, Any]]]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: np.ndarray of shape (dimension,)
            k: Number of results to return
            
        Returns:
            - distances: List of distances for top-k results
            - metadata: List of metadata dicts for top-k results
        """
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        distances, indices = self.index.search(query_embedding, k)
        
        distances = distances[0].tolist()
        results_metadata = [self.metadata[idx] for idx in indices[0]]
        
        return distances, results_metadata
    
    def save(self, index_path: str, metadata_path: str) -> None:
        """
        Save index and metadata to disk
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata pickle
        """
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        faiss.write_index(self.index, index_path)
        logger.info(f"✓ FAISS index saved to {index_path}")
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        logger.info(f"✓ Metadata saved to {metadata_path}")
    
    @staticmethod
    def load(index_path: str, metadata_path: str) -> 'FAISSVectorDB':
        """
        Load index and metadata from disk
        
        Args:
            index_path: Path to FAISS index
            metadata_path: Path to metadata pickle
            
        Returns:
            FAISSVectorDB instance with loaded data
        """
        index = faiss.read_index(index_path)
        logger.info(f"✓ FAISS index loaded from {index_path}")
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        logger.info(f"✓ Metadata loaded from {metadata_path}")
        
        db = FAISSVectorDB(dimension=index.d)
        db.index = index
        db.metadata = metadata
        
        return db


# ============================================================================
# MAIN PIPELINE
# ============================================================================
class Phase_A_Pipeline:
    """Main pipeline for Phase A: Cleaning -> Chunking -> Embedding -> Storage"""
    
    def __init__(self, config: Config = Config):
        self.config = config
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    def load_json(self, json_path: str) -> List[Dict[str, Any]]:
        """Load and parse JSON file"""
        logger.info(f"Loading JSON from: {json_path}")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            assessments = json.load(f)
        
        logger.info(f"✓ Loaded {len(assessments)} assessments from JSON")
        return assessments
    
    def clean_assessments(self, assessments: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        """
        Clean all assessments
        
        Returns:
            - cleaned_assessments: List of cleaned assessments
            - removed_count: Number of invalid assessments removed
        """
        logger.info("="*80)
        logger.info("STEP 1: DATA CLEANING")
        logger.info("="*80)
        
        cleaned = []
        invalid_count = 0
        
        for idx, assessment in enumerate(assessments):
            cleaned_assessment = DataCleaner.clean_assessment(assessment)
            
            if DataCleaner.validate_assessment(cleaned_assessment):
                cleaned.append(cleaned_assessment)
            else:
                invalid_count += 1
        
        logger.info(f"✓ Cleaned {len(cleaned)} assessments (removed {invalid_count} invalid)")
        
        # Save cleaned JSON
        with open(self.config.CLEANED_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Cleaned JSON saved to {self.config.CLEANED_JSON_PATH}")
        
        return cleaned, invalid_count
    
    def chunk_assessments(self, assessments: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Create chunks and metadata"""
        logger.info("="*80)
        logger.info("STEP 2: DATA CHUNKING")
        logger.info("="*80)
        
        chunks, metadata = DataChunker.chunk_assessments(assessments)
        
        # Save chunk info
        chunk_info = {
            'total_chunks': len(chunks),
            'chunk_dimension': self.config.EMBEDDING_DIM,
            'model_name': self.config.MODEL_NAME,
            'chunks': [
                {
                    'id': m['chunk_id'],
                    'name': m['assessment_name'],
                    'preview': chunks[m['chunk_id']][:100] + '...'
                }
                for m in metadata[:5]  # Save preview of first 5
            ]
        }
        
        with open(self.config.CHUNK_INFO_PATH, 'w', encoding='utf-8') as f:
            json.dump(chunk_info, f, indent=2)
        logger.info(f"✓ Chunk info saved to {self.config.CHUNK_INFO_PATH}")
        
        return chunks, metadata
    
    def generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for chunks"""
        logger.info("="*80)
        logger.info("STEP 3: EMBEDDING GENERATION")
        logger.info("="*80)
        
        generator = EmbeddingGenerator(model_name=self.config.MODEL_NAME)
        embeddings = generator.encode_batch(chunks, batch_size=self.config.BATCH_SIZE)
        
        return embeddings
    
    def build_vector_db(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> FAISSVectorDB:
        """Build and save FAISS index"""
        logger.info("="*80)
        logger.info("STEP 4: FAISS VECTOR DATABASE")
        logger.info("="*80)
        
        db = FAISSVectorDB(dimension=self.config.EMBEDDING_DIM)
        db.add_embeddings(embeddings, metadata)
        
        db.save(self.config.FAISS_INDEX_PATH, self.config.METADATA_PATH)
        
        return db
    
    def run(self, json_path: str = None):
        """Execute full Phase A pipeline"""
        json_path = json_path or self.config.INPUT_JSON_PATH
        
        logger.info("\n" + "="*80)
        logger.info("PHASE A: DATA INGESTION, CLEANING & EMBEDDING")
        logger.info("="*80 + "\n")
        
        # Step 1: Load JSON
        assessments = self.load_json(json_path)
        
        # Step 2: Clean assessments
        cleaned_assessments, removed_count = self.clean_assessments(assessments)
        
        # Validation
        if len(cleaned_assessments) < 377:
            logger.warning(f"⚠️  WARNING: Only {len(cleaned_assessments)} assessments after cleaning. Need minimum 377!")
        else:
            logger.info(f"✓ Assessment count: {len(cleaned_assessments)} (exceeds minimum 377)")
        
        # Step 3: Create chunks
        chunks, metadata = self.chunk_assessments(cleaned_assessments)
        
        # Step 4: Generate embeddings
        embeddings = self.generate_embeddings(chunks)
        
        # Step 5: Build FAISS index
        vector_db = self.build_vector_db(embeddings, metadata)
        
        logger.info("\n" + "="*80)
        logger.info("PHASE A COMPLETE")
        logger.info("="*80)
        logger.info(f"✓ Output directory: {self.config.OUTPUT_DIR}")
        logger.info(f"✓ FAISS index: {self.config.FAISS_INDEX_PATH}")
        logger.info(f"✓ Metadata: {self.config.METADATA_PATH}")
        logger.info(f"✓ Cleaned JSON: {self.config.CLEANED_JSON_PATH}")
        logger.info(f"✓ Chunk info: {self.config.CHUNK_INFO_PATH}")
        
        return vector_db, cleaned_assessments, metadata


# ============================================================================
# TESTING & DEMO
# ============================================================================
def demo_search(vector_db: FAISSVectorDB, generator: EmbeddingGenerator, query: str, k: int = 5):
    """
    Demo: Search for similar assessments
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"DEMO: Searching for: '{query}'")
    logger.info(f"{'='*80}")
    
    # Encode query
    query_embedding = generator.encode_single(query)
    
    # Search
    distances, results = vector_db.search(query_embedding, k=k)
    
    logger.info(f"Top {k} results:\n")
    for idx, (distance, result) in enumerate(zip(distances, results), 1):
        logger.info(f"{idx}. {result['assessment_name']}")
        logger.info(f"   Type: {result['test_type']}")
        logger.info(f"   Job Levels: {result['job_levels']}")
        logger.info(f"   Distance: {distance:.4f}")
        logger.info(f"   URL: {result['assessment_url']}\n")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import sys
    
    # Get JSON path from command line or use default
    json_file = sys.argv[1] if len(sys.argv) > 1 else Config.INPUT_JSON_PATH
    
    try:
        # Run Phase A pipeline
        pipeline = Phase_A_Pipeline()
        vector_db, cleaned_assessments, metadata = pipeline.run(json_file)
        
        # Demo search
        generator = EmbeddingGenerator()
        demo_queries = [
            "Java developer with team collaboration skills",
            "Python and SQL database engineer",
            "Cognitive ability and personality assessment"
        ]
        
        for query in demo_queries:
            demo_search(vector_db, generator, query, k=5)
        
        logger.info("\n✅ Phase A Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        sys.exit(1)