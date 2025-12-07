from sqlalchemy.orm import Session
from app.models.photo import Photo
from app.services.ai_service import AIService
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class VectorService:
    def __init__(self, db: Session):
        self.db = db
        self.ai_service = AIService()

    def search_similar_photos(self, query: str, limit: int = 12) -> List[Dict]:
        """
        Search for photos using semantic similarity (PGVector).
        Uses AIService to generate embeddings for the query.
        """
        # 1. Generate embedding for the query text
        query_embedding = self.ai_service.generate_embedding(query)

        if not query_embedding:
            logger.warning("Could not generate embedding for query")
            return []

        # 2. Search in PostgeSQL using pgvector cosine distance
        # Note: We use the <-> operator aka cosine_distance in SQLModel/SQLAlchemy
        try:
            results = self.db.query(Photo).order_by(
                Photo.embedding.cosine_distance(query_embedding)
            ).limit(limit).all()

            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
