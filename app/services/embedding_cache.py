import redis
import os
import json
import hashlib
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class EmbeddingCache:
    """
    Cache inteligente para embeddings e resultados de busca usando Redis.
    Implementa cache com TTL para melhorar performance das buscas.
    """

    def __init__(self, redis_client=None):
        """Initialize cache with Redis client"""
        self.redis = redis_client or redis.from_url(os.getenv('REDIS_URL', 'redis://redis:6379'))
        self.ttl = 3600 * 24  # 24 horas
        self.embedding_ttl = 3600 * 24 * 7  # 7 dias para embeddings
        self.search_ttl = 3600 * 6  # 6 horas para resultados de busca

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding for text

        Args:
            text: Text to get embedding for

        Returns:
            Cached embedding or None if not found
        """
        try:
            key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
            cached = self.redis.get(key)
            if cached:
                logger.debug(f"Cache hit for embedding: {text[:50]}...")
                return json.loads(cached)
            logger.debug(f"Cache miss for embedding: {text[:50]}...")
            return None
        except Exception as e:
            logger.warning(f"Error getting cached embedding: {e}")
            return None

    def set_embedding(self, text: str, embedding: List[float]):
        """
        Cache embedding for text

        Args:
            text: Text to cache embedding for
            embedding: Embedding vector to cache
        """
        try:
            key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
            self.redis.setex(key, self.embedding_ttl, json.dumps(embedding))
            logger.debug(f"Cached embedding for: {text[:50]}...")
        except Exception as e:
            logger.warning(f"Error caching embedding: {e}")

    def get_search_results(self, query_hash: str) -> Optional[List[Dict]]:
        """
        Get cached search results

        Args:
            query_hash: Hash of the search query

        Returns:
            Cached search results or None if not found
        """
        try:
            key = f"search:{query_hash}"
            cached = self.redis.get(key)
            if cached:
                logger.debug(f"Cache hit for search: {query_hash}")
                return json.loads(cached)
            logger.debug(f"Cache miss for search: {query_hash}")
            return None
        except Exception as e:
            logger.warning(f"Error getting cached search results: {e}")
            return None

    def set_search_results(self, query_hash: str, results: List[Dict]):
        """
        Cache search results

        Args:
            query_hash: Hash of the search query
            results: Search results to cache
        """
        try:
            key = f"search:{query_hash}"
            self.redis.setex(key, self.search_ttl, json.dumps(results))
            logger.debug(f"Cached search results for: {query_hash}")
        except Exception as e:
            logger.warning(f"Error caching search results: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring

        Returns:
            Dictionary with cache statistics
        """
        try:
            # Get Redis info
            info = self.redis.info()

            # Count keys by pattern (approximate)
            embedding_keys = len(self.redis.keys("embedding:*"))
            search_keys = len(self.redis.keys("search:*"))

            return {
                "total_keys": info.get('db0', {}).get('keys', 0),
                "embedding_keys": embedding_keys,
                "search_keys": search_keys,
                "memory_used": info.get('used_memory_human', 'unknown'),
                "uptime": info.get('uptime_in_seconds', 0),
                "connected_clients": info.get('connected_clients', 0)
            }
        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}")
            return {
                "error": str(e),
                "status": "error"
            }

    def clear_cache(self, pattern: str = "*"):
        """
        Clear cache entries matching pattern

        Args:
            pattern: Pattern to match (default: "*" for all)
        """
        try:
            keys = self.redis.keys(f"{pattern}:*")
            if keys:
                self.redis.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache entries with pattern: {pattern}")
            else:
                logger.info(f"No cache entries found with pattern: {pattern}")
        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")

    def health_check(self) -> bool:
        """
        Check if Redis cache is healthy

        Returns:
            True if healthy, False otherwise
        """
        try:
            self.redis.ping()
            return True
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return False
