from sqlalchemy.orm import Session
from app.models.photo import Photo
from app.services.ai_service import AIService
from app.services.embedding_cache import EmbeddingCache
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class VectorService:
    def __init__(self, db: Session):
        self.db = db
        self.ai_service = AIService()
        self.cache = EmbeddingCache()

    def search_similar_photos(self, query: str, limit: int = 12) -> List[Dict]:
        """
        Search for photos using semantic similarity (PGVector).
        Uses AIService to generate embeddings for the query with caching.
        """
        try:
            # 1. Try to get cached embedding first
            query_embedding = self.cache.get_embedding(query)

            if not query_embedding:
                # Generate new embedding if not cached
                query_embedding = self.ai_service.generate_embedding(query)
                if not query_embedding:
                    logger.warning("Could not generate embedding for query")
                    return []
                # Cache the new embedding
                self.cache.set_embedding(query, query_embedding)

            # 2. Search in PostgreSQL using pgvector cosine distance
            # Note: We use the <-> operator aka cosine_distance in SQLModel/SQLAlchemy
            results = self.db.query(Photo).order_by(
                Photo.embedding.cosine_distance(query_embedding)
            ).limit(limit).all()

            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
