"""
Test imports work correctly.
"""

def test_imports():
    from hybrid_rag import config, models
    from hybrid_rag.chunkers import FixedChunker, HeadingChunker
    from hybrid_rag.ingestion import clean_markdown, IngestionPipeline
    from hybrid_rag.retrieval import hybrid_search, QdrantClient
    from hybrid_rag.generation import synthesize_answer
    from hybrid_rag.evaluation import compute_retrieval_metrics, closed_book_answer
    assert True