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
        Returns list of dicts with 'photo' and 'similarity_score'.
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
            # Query both Photo and distance
            from sqlalchemy import func
            results = self.db.query(
                Photo,
                Photo.embedding.cosine_distance(query_embedding).label('distance')
            ).order_by(
                Photo.embedding.cosine_distance(query_embedding)
            ).limit(limit).all()

            # Format results with similarity score (1 - distance, since cosine_distance = 1 - cos_sim)
            formatted_results = []
            for photo, distance in results:
                similarity_score = 1.0 - distance  # Convert distance to similarity
                formatted_results.append({
                    'photo_id': photo.id,
                    'similarity': similarity_score
                })

            return formatted_results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def search_similar_photos_by_embedding(self, query_embedding: List[float], limit: int = 12) -> List[Dict]:
        """
        Search for photos using a pre-computed embedding vector.
        Returns list of dicts with 'photo_id' and 'similarity'.
        """
        try:
            # Search in PostgreSQL using pgvector cosine distance
            from sqlalchemy import func
            results = self.db.query(
                Photo,
                Photo.image_embedding.cosine_distance(query_embedding).label('distance')
            ).filter(
                Photo.image_embedding.isnot(None)  # Only photos with image embeddings
            ).order_by(
                Photo.image_embedding.cosine_distance(query_embedding)
            ).limit(limit).all()

            # Format results with similarity score
            formatted_results = []
            for photo, distance in results:
                similarity_score = 1.0 - distance
                formatted_results.append({
                    'photo_id': photo.id,
                    'similarity': similarity_score
                })

            return formatted_results
        except Exception as e:
            logger.error(f"Vector search by embedding failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
