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
        Returns list of dicts with 'photo_id', 'similarity', and 'justification'.
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

            # Minimum similarity threshold for results
            min_similarity_threshold = 0.3  # Lower threshold since we have relevance boosting

            # 2. Search in PostgreSQL using pgvector cosine distance
            # Get more results initially for better re-ranking
            initial_limit = min(limit * 3, 50)  # Get up to 3x more results for re-ranking
            results = self.db.query(
                Photo,
                Photo.embedding.cosine_distance(query_embedding).label('distance')
            ).order_by(
                Photo.embedding.cosine_distance(query_embedding)
            ).limit(initial_limit).all()

            # Format results with intelligent re-ranking
            filtered_results = []
            for photo, distance in results:
                similarity_score = max(-1.0, min(1.0, 1.0 - distance))

                # Calculate relevance boost based on multiple factors
                relevance_boost = self._calculate_relevance_boost(query, photo.description or "", photo.original_filename)
                final_score = similarity_score + relevance_boost

                # Include results with reasonable final score
                if final_score >= min_similarity_threshold:
                    justification = self._generate_justification(query, photo.description or "", photo.original_filename, similarity_score, relevance_boost)
                    filtered_results.append({
                        'photo_id': photo.id,
                        'similarity': final_score,
                        'justification': justification
                    })

            # Sort filtered results by final similarity score
            filtered_results.sort(key=lambda x: x['similarity'], reverse=True)

            return filtered_results[:limit]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _calculate_relevance_boost(self, query: str, description: str, filename: str) -> float:
        """
        Calculate relevance boost based on keyword matches in filename and description.
        Returns a boost value between 0.0 and 1.0.
        """
        try:
            query_lower = query.lower()
            desc_lower = description.lower()
            filename_lower = filename.lower()

            boost = 0.0

            # Define keyword groups for common searches
            keyword_groups = {
                'animals': {
                    'dogs': ['cachorro', 'cão', 'dog', 'cachorrinho'],
                    'cats': ['gato', 'felino', 'cat', 'gatinho', 'gata', 'kitty', 'feline'],
                    'birds': ['pássaro', 'ave', 'bird'],
                    'horses': ['cavalo', 'horse', 'égua']
                },
                'colors': ['preto', 'branco', 'marrom', 'cinza', 'laranja', 'amarelo', 'vermelho', 'azul', 'verde'],
                'emotions': ['feliz', 'alegre', 'triste', 'brincando', 'dormindo', 'comendo'],
                'environments': ['praia', 'cidade', 'natureza', 'jardim', 'quintal', 'rua']
            }

            # Check filename matches (higher weight)
            for category, keywords in keyword_groups.items():
                if isinstance(keywords, dict):
                    for subcat, subkeywords in keywords.items():
                        if any(kw in filename_lower for kw in subkeywords):
                            if any(kw in query_lower for kw in subkeywords):
                                boost += 0.4  # Strong boost for filename + query match
                            else:
                                boost += 0.2  # Moderate boost for filename match
                else:
                    if any(kw in filename_lower for kw in keywords):
                        if any(kw in query_lower for kw in keywords):
                            boost += 0.3

            # Check description matches (lower weight)
            for category, keywords in keyword_groups.items():
                if isinstance(keywords, dict):
                    for subcat, subkeywords in keywords.items():
                        if any(kw in desc_lower for kw in subkeywords):
                            if any(kw in query_lower for kw in subkeywords):
                                boost += 0.2  # Boost for description + query match
                else:
                    if any(kw in desc_lower for kw in keywords):
                        if any(kw in query_lower for kw in keywords):
                            boost += 0.1

            return min(boost, 1.0)  # Cap at 1.0

        except Exception as e:
            logger.error(f"Error calculating relevance boost: {e}")
            return 0.0

    def _generate_justification(self, query: str, description: str, filename: str, similarity_score: float, relevance_boost: float) -> str:
        """
        Generate a human-readable justification for why this photo was selected.
        """
        try:
            reasons = []

            if similarity_score > 0.6:
                reasons.append("High semantic similarity")
            elif similarity_score > 0.4:
                reasons.append("Moderate semantic similarity")

            if relevance_boost > 0.3:
                reasons.append("Strong keyword match")
            elif relevance_boost > 0.1:
                reasons.append("Keyword match")

            # Check for specific matches
            query_lower = query.lower()
            filename_lower = filename.lower()

            if 'gato' in query_lower and ('gato' in filename_lower or 'cat' in filename_lower):
                reasons.append("Filename contains 'gato'/'cat'")
            elif 'cachorro' in query_lower and ('cachorro' in filename_lower or 'dog' in filename_lower):
                reasons.append("Filename contains 'cachorro'/'dog'")

            if not reasons:
                reasons.append("Semantic similarity match")

            return "; ".join(reasons)

        except Exception as e:
            logger.error(f"Error generating justification: {e}")
            return "Match found"

        except Exception as e:
            logger.error(f"Error generating justification: {e}")
            return "Match found"
        """
        Search for photos using a pre-computed embedding vector.
        Returns list of dicts with 'photo_id', 'similarity', and 'justification'.
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
                justification = f"Visual similarity: {similarity_score:.3f} (based on CLIP image embeddings)"
                formatted_results.append({
                    'photo_id': photo.id,
                    'similarity': similarity_score,
                    'justification': justification
                })

            return formatted_results
        except Exception as e:
            logger.error(f"Vector search by embedding failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
