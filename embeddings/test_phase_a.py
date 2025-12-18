
import json
import logging
from building_embedding import (
    Config, DataCleaner, DataChunker, EmbeddingGenerator,
    FAISSVectorDB, Phase_A_Pipeline
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# TEST SUITE
# ============================================================================

def test_data_cleaner():
    """Test data cleaning functionality"""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: DATA CLEANER")
    logger.info("="*80)
    
    test_cases = [
        {
            "input": {
                "Assessment Name": "  Python Programming  ",
                "Job Levels": "Mid-Professional, Senior,",
                "Languages": "English (USA),",
                "Description": "  Multi-choice  test  that measures..."
            },
            "expected": {
                "Assessment Name": "Python Programming",
                "Job Levels": "Mid-Professional, Senior",
                "Languages": "English (USA)",
                "Description": "Multi-choice test that measures..."
            }
        }
    ]
    
    for idx, test in enumerate(test_cases, 1):
        result = DataCleaner.clean_assessment(test["input"])
        
        # Check specific fields
        assert result["Assessment Name"] == test["expected"]["Assessment Name"], \
            f"Name mismatch: {result['Assessment Name']}"
        assert result["Job Levels"] == test["expected"]["Job Levels"], \
            f"Job Levels mismatch: {result['Job Levels']}"
        
        logger.info(f"✓ Test case {idx} passed")
    
    logger.info("✓ All cleaning tests passed!")


def test_data_chunker():
    """Test data chunking functionality"""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: DATA CHUNKER")
    logger.info("="*80)
    
    assessment = {
        "Assessment Name": "Python Programming",
        "Assessment URL": "https://www.shl.com/python",
        "Description": "Test for Python knowledge",
        "Test Type": "Knowledge & Skills",
        "Job Levels": "Mid-Professional",
        "Languages": "English (USA)",
        "Time (Minutes)": "20",
        "Remote Testing Support": "Yes"
    }
    
    chunk = DataChunker.create_chunk(assessment)
    
    # Verify chunk contains key information
    assert "Python Programming" in chunk, "Assessment name not in chunk"
    assert "Test for Python knowledge" in chunk, "Description not in chunk"
    assert "20 minutes" in chunk, "Time not in chunk"
    
    logger.info(f"Created chunk (length: {len(chunk)} chars):")
    logger.info(f"Preview: {chunk[:150]}...")
    logger.info("✓ Chunk creation test passed!")


def test_embedding_generator():
    """Test embedding generation"""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: EMBEDDING GENERATOR")
    logger.info("="*80)
    
    logger.info("Initializing embedding model...")
    generator = EmbeddingGenerator(model_name=Config.MODEL_NAME)
    
    # Test single embedding
    text = "Python developer with strong communication skills"
    embedding = generator.encode_single(text)
    
    assert embedding.shape == (Config.EMBEDDING_DIM,), \
        f"Embedding shape mismatch: {embedding.shape}"
    assert embedding.dtype == "float32", f"Embedding dtype: {embedding.dtype}"
    
    logger.info(f"✓ Generated single embedding (shape: {embedding.shape})")
    logger.info(f"  Norm: {(embedding**2).sum()**.5:.4f}")
    
    # Test batch embedding
    texts = [
        "Java developer",
        "Python developer",
        "Frontend engineer"
    ]
    embeddings = generator.encode_batch(texts, batch_size=2)
    
    assert embeddings.shape == (len(texts), Config.EMBEDDING_DIM), \
        f"Batch shape mismatch: {embeddings.shape}"
    
    logger.info(f"✓ Generated batch embeddings (shape: {embeddings.shape})")
    
    # Test similarity
    similarity_java_python = (embeddings[0] * embeddings[1]).sum()
    similarity_java_frontend = (embeddings[0] * embeddings[2]).sum()
    
    logger.info(f"  Similarity (Java vs Python): {similarity_java_python:.4f}")
    logger.info(f"  Similarity (Java vs Frontend): {similarity_java_frontend:.4f}")
    logger.info("✓ Embedding generation tests passed!")


def test_faiss_vector_db():
    """Test FAISS vector database"""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: FAISS VECTOR DATABASE")
    logger.info("="*80)
    
    import numpy as np
    
    # Create test data
    embeddings = np.random.rand(10, Config.EMBEDDING_DIM).astype(np.float32)
    metadata = [
        {
            'chunk_id': i,
            'assessment_name': f'Assessment {i}',
            'assessment_url': f'https://example.com/{i}',
            'test_type': 'Test',
            'job_levels': 'Senior',
            'description': f'Test assessment {i}',
            'time_minutes': 20,
            'remote_support': 'Yes'
        }
        for i in range(10)
    ]
    
    # Create DB and add embeddings
    db = FAISSVectorDB()
    db.add_embeddings(embeddings, metadata)
    
    assert db.index.ntotal == 10, f"DB should have 10 items, has {db.index.ntotal}"
    logger.info(f"✓ Added 10 embeddings to FAISS index")
    
    # Test search
    query_embedding = embeddings[0].copy()
    distances, results = db.search(query_embedding, k=5)
    
    assert len(results) == 5, f"Should return 5 results, got {len(results)}"
    assert results[0]['assessment_name'] == 'Assessment 0', "First result should be exact match"
    
    logger.info(f"✓ Search returned {len(results)} results")
    logger.info(f"  Top result: {results[0]['assessment_name']} (distance: {distances[0]:.4f})")
    logger.info("✓ FAISS database tests passed!")


def test_pipeline_integration():
    """Test full pipeline with sample data"""
    logger.info("\n" + "="*80)
    logger.info("TEST 5: FULL PIPELINE INTEGRATION")
    logger.info("="*80)
    
    # Create sample JSON
    sample_data = [
        {
            "Assessment Name": "Python Programming",
            "Assessment URL": "https://www.shl.com/python",
            "Remote Testing Support": "Yes",
            "Adaptive/IRT Support": "No",
            "Test Type": "Knowledge & Skills",
            "Test Type Keys": ["K"],
            "Description": "Test for Python knowledge covering basics, OOP, and libraries.",
            "Time (Minutes)": "20",
            "Job Levels": "Mid-Professional, Senior",
            "Languages": "English (USA)"
        },
        {
            "Assessment Name": "Communication Skills",
            "Assessment URL": "https://www.shl.com/communication",
            "Remote Testing Support": "Yes",
            "Adaptive/IRT Support": "No",
            "Test Type": "Personality & Behavior",
            "Test Type Keys": ["P"],
            "Description": "Assessment of verbal and written communication abilities.",
            "Time (Minutes)": "15",
            "Job Levels": "General Population",
            "Languages": "English (USA)"
        },
        {
            "Assessment Name": "Leadership Potential",
            "Assessment URL": "https://www.shl.com/leadership",
            "Remote Testing Support": "No",
            "Adaptive/IRT Support": "No",
            "Test Type": "Personality & Behavior",
            "Test Type Keys": ["P"],
            "Description": "Evaluates leadership capability and management potential.",
            "Time (Minutes)": None,
            "Job Levels": "Manager, Executive",
            "Languages": "English (USA)"
        }
    ]
    
    # Save sample JSON
    with open("test_assessments.json", 'w') as f:
        json.dump(sample_data, f, indent=2)
    logger.info("✓ Created test JSON file")
    
    # Run pipeline
    pipeline = Phase_A_Pipeline()
    try:
        vector_db, cleaned, metadata = pipeline.run("test_assessments.json")
        
        assert len(cleaned) == 3, f"Should have 3 cleaned assessments, got {len(cleaned)}"
        assert len(metadata) == 3, f"Should have 3 metadata entries, got {len(metadata)}"
        assert vector_db.index.ntotal == 3, f"DB should have 3 items, got {vector_db.index.ntotal}"
        
        logger.info(f"✓ Cleaned and processed {len(cleaned)} assessments")
        logger.info(f"✓ Created {len(metadata)} metadata entries")
        logger.info(f"✓ Built FAISS index with {vector_db.index.ntotal} embeddings")
        
        # Test search
        generator = EmbeddingGenerator()
        query = "Python developer with leadership skills"
        query_embedding = generator.encode_single(query)
        distances, results = vector_db.search(query_embedding, k=2)
        
        logger.info(f"\nTest Search: '{query}'")
        logger.info(f"Results:")
        for i, (dist, result) in enumerate(zip(distances, results), 1):
            logger.info(f"  {i}. {result['assessment_name']} (distance: {dist:.4f})")
        
        logger.info("✓ Pipeline integration test passed!")
        
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}", exc_info=True)
        raise


# ============================================================================
# MAIN
# ============================================================================

def run_all_tests():
    """Run all tests"""
    logger.info("\n" + "="*80)
    logger.info("PHASE A TEST SUITE")
    logger.info("="*80)
    
    tests = [
        ("Data Cleaner", test_data_cleaner),
        ("Data Chunker", test_data_chunker),
        ("Embedding Generator", test_embedding_generator),
        ("FAISS Vector DB", test_faiss_vector_db),
        ("Pipeline Integration", test_pipeline_integration),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            logger.error(f"❌ {test_name} test failed: {e}", exc_info=True)
            failed += 1
    
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    logger.info(f"✓ Passed: {passed}/{len(tests)}")
    logger.info(f"❌ Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        logger.info("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        logger.error(f"\n❌ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = run_all_tests()
    sys.exit(exit_code)