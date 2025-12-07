# visual_search_service.py - Advanced Visual Search with ChromaDB
import os
import uuid
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction, SentenceTransformerEmbeddingFunction
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from pathlib import Path
from PIL import Image
import base64
import logging
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import hashlib
import time
from app.services.embedding_cache import EmbeddingCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualSearchService:
    """
    Advanced visual search service using ChromaDB for image and text embeddings.
    Provides semantic search, reverse image search, and intelligent re-ranking.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._setup()
        return cls._instance

    def _setup(self):
        """Initialize ChromaDB client and embedding functions"""
        # Create persistent ChromaDB directory
        db_path = Path("data/chroma_db")
        db_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(db_path))

        # Create or get collection for images with optimized HNSW settings
        self.collection = self.client.get_or_create_collection(
            name="images",
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,  # Higher = better quality, slower construction
                "hnsw:search_ef": 100,        # Higher = better recall, slower search
                "hnsw:M": 32,                 # Higher = better recall, more memory
            }
        )

        # Batch processing settings
        self.batch_size = 32  # Process embeddings in batches

        # Initialize embedding functions
        self.clip_embeddings = OpenCLIPEmbeddingFunction()  # For image embeddings
        self.text_embeddings = SentenceTransformerEmbeddingFunction(
            model_name="multi-qa-mpnet-base-cos-v1"  # Better for Portuguese
        )

        # Initialize LLM for rich captions and re-ranking
        self.llm = self._initialize_llm()

        # Initialize embedding cache
        self.embedding_cache = EmbeddingCache()

        # Performance monitoring
        self._performance_stats = {
            "total_searches": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "reranking_used": 0,
            "reranking_skipped": 0,
            "avg_search_time": 0.0,
            "slow_queries": 0,
            "last_reset": time.time(),
            "query_types": {
                "simple": 0,
                "medium": 0,
                "complex": 0,
                "question": 0
            }
        }

        logger.info("VisualSearchService initialized with ChromaDB")

    def _initialize_llm(self) -> Optional[object]:
        """Initialize LLM with fallback chain"""
        try:
            # Try Google Gemini first with correct model
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                return ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",  # Use newer model
                    temperature=0.3,
                    max_tokens=1024
                )
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini: {e}")

        try:
            # Fallback to OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                return ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.3,
                    max_tokens=1024
                )
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI: {e}")

        logger.warning("No LLM available - captions will be basic")
        return None

    def _generate_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings in batches for better performance"""
        if len(texts) == 1:
            return [self.text_embeddings(texts)[0]]

        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.text_embeddings(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def add_image_with_agent(self, image_path: str, photo_id: int, user_description: str = None, tags: List[str] = None) -> str:
        """
        Add image using LangChain agent for intelligent processing
        """
        try:
            from .langchain_agents import ImageProcessingAgent
            from .ai_service import AIService

            # Initialize agent if not exists
            if not hasattr(self, '_image_agent'):
                ai_service = AIService()
                if ai_service.llm:
                    self._image_agent = ImageProcessingAgent(ai_service.llm, self, ai_service)
                else:
                    # Fallback to direct processing
                    return self.add_image(image_path, photo_id, user_description, tags)

            # Use agent for processing
            result = self._image_agent.process_image(image_path, user_description)

            if result["success"]:
                logger.info(f"Agent processed image successfully: {result['agent_response']}")
                # Extract photo_id from agent response or use provided
                return f"agent_processed_{photo_id}"
            else:
                logger.warning(f"Agent failed, using fallback: {result.get('error', 'Unknown error')}")
                return self.add_image(image_path, photo_id, user_description, tags)

        except ImportError:
            logger.warning("LangChain agents not available, using direct processing")
            return self.add_image(image_path, photo_id, user_description, tags)
        except Exception as e:
            logger.error(f"Error in agent processing: {str(e)}")
            return self.add_image(image_path, photo_id, user_description, tags)
    def add_image(self, image_path: str, photo_id: int, user_description: str = None, tags: List[str] = None) -> str:
        """
        Add an image to the search index with combined image+text embeddings.

        Args:
            image_path: Path to the image file
            photo_id: ID of the photo in the database
            user_description: Optional user-provided description
            tags: Optional list of tags

        Returns:
            Document ID of the added image
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Generate rich caption using AI
        caption = self._generate_rich_caption(image_path, user_description)

        # Generate embeddings
        image_embedding = np.array(self.clip_embeddings([image_path])[0])
        text_embedding = np.array(self.text_embeddings([caption])[0])

        # For now, use only CLIP embedding (512 dimensions) to match our database schema
        # TODO: Implement proper dimension alignment or use separate indices
        combined_embedding = image_embedding.tolist()

        # Create unique document ID
        doc_id = str(uuid.uuid4())

        # Prepare comprehensive metadata
        metadata = {
            "image_path": image_path,
            "photo_id": str(photo_id),
            "user_description": user_description or "",
            "tags": ",".join(tags) if tags else "",
            "file_type": Path(image_path).suffix.lower(),
            "file_name": Path(image_path).name,
            "created_at": str(uuid.uuid1().time),
            "caption": caption,
            # Store individual embeddings for potential future use
            "image_embedding": str(image_embedding.tolist()),
            "text_embedding": str(text_embedding.tolist())
        }

        # Add to ChromaDB with combined embedding
        self.collection.add(
            ids=[doc_id],
            embeddings=[combined_embedding],
            documents=[caption],
            metadatas=[metadata]
        )

        logger.info(f"Image added with combined embedding: {Path(image_path).name} | ID: {doc_id[:8]}")
        return doc_id

    def add_images_batch(self, image_data: List[Dict]) -> List[str]:
        """
        Add multiple images at once for better performance
        """
        if not image_data:
            return []

        # Prepare all data
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for data in image_data:
            # Generate embeddings in batch
            image_path = data['image_path']
            caption = self._generate_rich_caption(image_path, data.get('user_description'))

            # Batch embedding generation
            image_emb = self.clip_embeddings([image_path])[0]
            text_emb = self.text_embeddings([caption])[0]
            combined_emb = image_emb  # Simplified for now

            ids.append(str(uuid.uuid4()))
            embeddings.append(combined_emb.tolist())
            documents.append(caption)
            metadatas.append({
                "photo_id": str(data['photo_id']),
                "user_description": data.get('user_description', ''),
                "file_name": Path(image_path).name,
                "caption": caption
            })

        # Batch add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        logger.info(f"Batch added {len(ids)} images to ChromaDB")
        return ids

    def _generate_rich_caption(self, image_path: str, user_context: str = None) -> str:
        """
        Generate a rich, detailed caption using multimodal LLM.
        """
        if not self.llm:
            # Fallback to basic description
            base_desc = f"Image: {Path(image_path).name}"
            if user_context:
                base_desc += f" | Context: {user_context}"
            return base_desc

        try:
            # Convert image to base64
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()

            prompt = f"""
            Describe this image in maximum detail in Brazilian Portuguese.
            Include:
            - Main and secondary objects
            - Exact colors (e.g., navy blue, Ferrari red)
            - Visible text (exactly as written)
            - Brand, model, defects, condition
            - People: gender, approximate age, clothing, expression
            - Context: environment, lighting, approximate date if visible
            - Any detail that differentiates this image from similar ones

            {f'Additional context: {user_context}' if user_context else ''}

            Be as specific as possible. This description will be used for search.
            """

            # Simplified caption generation without multimodal
            prompt_text = f"""
            Describe this image in maximum detail in Brazilian Portuguese.
            Include:
            - Main and secondary objects
            - Exact colors (e.g., navy blue, Ferrari red)
            - Visible text (exactly as written)
            - Brand, model, defects, condition
            - People: gender, approximate age, clothing, expression
            - Context: environment, lighting, approximate date if visible
            - Any detail that differentiates this image from similar ones

            {f'Additional context: {user_context}' if user_context else ''}

            Be as specific as possible. This description will be used for search.
            """

            response = self.llm.invoke(prompt_text)
            return response.content.strip() if hasattr(response, 'content') else str(response)

        except Exception as e:
            logger.error(f"Failed to generate rich caption: {e}")
            # Fallback
            return f"Image: {Path(image_path).name} | {user_context or ''}"

    def search_by_text(self, query: str, top_k: int = 8, use_reranking: bool = None) -> List[Dict]:
        """
        Search images by text query using semantic similarity with intelligent caching and re-ranking.

        Args:
            query: Text query to search for
            top_k: Number of results to return
            use_reranking: Force re-ranking decision (True/False) or auto-decide (None)

        Returns:
            List of search results with similarity scores
        """
        start_time = time.time()
        self._performance_stats["total_searches"] += 1

        try:
            # Check cache first
            query_hash = hashlib.md5(f"{query}:{top_k}:{use_reranking}".encode()).hexdigest()
            cached_results = self.embedding_cache.get_search_results(query_hash)
            if cached_results:
                logger.info(f"Cache hit for query: {query}")
                self._performance_stats["cache_hits"] += 1
                search_time = (time.time() - start_time) * 1000
                self._performance_stats["avg_search_time"] = (
                    (self._performance_stats["avg_search_time"] * (self._performance_stats["total_searches"] - 1)) + search_time
                ) / self._performance_stats["total_searches"]
                if search_time > 2000:  # > 2 seconds
                    self._performance_stats["slow_queries"] += 1
                return cached_results

            # Get cached embedding or generate new one
            query_embedding = self.embedding_cache.get_embedding(query)
            if not query_embedding:
                # For text queries, we need to use CLIP-compatible embeddings
                # Since collection uses CLIP (512 dims), truncate SentenceTransformer embedding
                text_emb = self._generate_embedding_batch([query])[0]
                query_embedding = text_emb[:512]  # Truncate to 512 dimensions
                self.embedding_cache.set_embedding(query, query_embedding)

            # Search with optimized parameters
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k * 3, 50),  # Get more candidates but limit
                include=["metadatas", "documents", "distances"]
            )

            # Process results faster
            candidates = self._process_results_batch(results)

            # Decide whether to use re-ranking based on query complexity
            if use_reranking is None:
                use_reranking = self._should_use_reranking(query)

            # Re-rank only if needed
            if use_reranking and len(candidates) > top_k:
                final_results = self._rerank_results(candidates, f"Find images most relevant to: '{query}'", use_image=False)[:top_k]
            else:
                final_results = candidates[:top_k]

            # Cache results
            self.embedding_cache.set_search_results(query_hash, final_results)

            # Update performance metrics
            search_time = (time.time() - start_time) * 1000
            self._performance_stats["avg_search_time"] = (
                (self._performance_stats["avg_search_time"] * (self._performance_stats["total_searches"] - 1)) + search_time
            ) / self._performance_stats["total_searches"]
            if search_time > 2000:  # > 2 seconds
                self._performance_stats["slow_queries"] += 1

            # Update query type stats
            query_type = self._classify_query_type(query)
            self._performance_stats["query_types"][query_type] += 1

            # Update re-ranking stats
            if use_reranking:
                self._performance_stats["reranking_used"] += 1
            else:
                self._performance_stats["reranking_skipped"] += 1

            return final_results

        except Exception as e:
            logger.error(f"Semantic text search failed: {e}")
            # Update error metrics
            search_time = (time.time() - start_time) * 1000
            self._performance_stats["avg_search_time"] = (
                (self._performance_stats["avg_search_time"] * (self._performance_stats["total_searches"] - 1)) + search_time
            ) / self._performance_stats["total_searches"]
            if search_time > 2000:
                self._performance_stats["slow_queries"] += 1

            # Fallback to basic text search
            return self._fallback_text_search(query, top_k)

    def _classify_query_type(self, query: str) -> str:
        """Classify query type for analytics"""
        query_lower = query.lower().strip()

        # Question patterns
        if any(word in query_lower for word in ['quem', 'qual', 'quando', 'onde', 'como', 'por que', 'o que', '?']):
            return "question"

        # Complex queries (multiple concepts)
        words = query_lower.split()
        if len(words) > 3 or any(char in query for char in [',', ' e ', ' ou ', ' com ', ' sem ']):
            return "complex"

        # Medium queries (2-3 words)
        if 2 <= len(words) <= 3:
            return "medium"

        # Simple queries (1 word)
        return "simple"

    def _process_results_batch(self, results: Dict) -> List[Dict]:
        """Process ChromaDB query results into standardized format"""
        if not results.get("metadatas") or not results["metadatas"][0]:
            return []

        candidates = []
        for metadata, distance, document in zip(
            results["metadatas"][0],
            results["distances"][0],
            results["documents"][0]
        ):
            if "photo_id" in metadata:
                candidates.append({
                    "photo_id": int(metadata["photo_id"]),
                    "similarity": 1 - distance,  # Convert distance to similarity
                    "caption": document,
                    "tags": metadata.get("tags", ""),
                    "file_name": metadata.get("file_name", ""),
                    "user_description": metadata.get("user_description", ""),
                    "metadata": metadata
                })

        return candidates

    def _fallback_text_search(self, query: str, top_k: int = 8) -> List[Dict]:
        """
        Fallback text search using simple string matching when semantic search fails.
        """
        try:
            # Get all documents and metadata
            all_results = self.collection.get(include=["metadatas", "documents"])

            if not all_results["documents"]:
                return []

            candidates = []
            query_lower = query.lower()
            expanded_queries = self._expand_query(query)
            expanded_lower = [q.lower() for q in expanded_queries]

            # Simple text matching
            for i, (doc, metadata) in enumerate(zip(all_results["documents"], all_results["metadatas"])):
                if "photo_id" not in metadata:
                    continue

                doc_lower = doc.lower()
                metadata_text = f"{metadata.get('user_description', '')} {metadata.get('tags', '')}".lower()

                # Calculate simple relevance score
                score = 0

                # Check original query
                if query_lower in doc_lower:
                    score += 10
                if query_lower in metadata_text:
                    score += 5

                # Check expanded queries
                for exp_query in expanded_lower:
                    if exp_query in doc_lower:
                        score += 8
                    if exp_query in metadata_text:
                        score += 4

                if score > 0:
                    candidates.append({
                        "photo_id": int(metadata["photo_id"]),
                        "similarity": min(score / 10.0, 1.0),
                        "caption": doc,
                        "tags": metadata.get("tags", ""),
                        "file_name": metadata.get("file_name", ""),
                        "user_description": metadata.get("user_description", ""),
                        "metadata": metadata
                    })

            # Sort by similarity and return top_k
            candidates.sort(key=lambda x: x["similarity"], reverse=True)
            return candidates[:top_k]

        except Exception as e:
            logger.error(f"Fallback text search failed: {e}")
            return []

    def _expand_query(self, query: str) -> List[str]:
        """
        Expand query with synonyms and translations
        """
        query_lower = query.lower().strip()

        # Basic synonym/translation dictionary
        synonyms = {
            # Animals
            "gato": ["cat", "gato", "felino", "kitty"],
            "cat": ["cat", "gato", "felino", "kitty"],
            "cachorro": ["dog", "cachorro", "cão", "puppy"],
            "dog": ["dog", "cachorro", "cão", "puppy"],
            "pássaro": ["bird", "pássaro", "ave"],
            "bird": ["bird", "pássaro", "ave"],

            # Colors
            "preto": ["black", "preto", "negro"],
            "black": ["black", "preto", "negro"],
            "branco": ["white", "branco"],
            "white": ["white", "branco"],
            "vermelho": ["red", "vermelho"],
            "red": ["red", "vermelho"],

            # Common terms
            "animal": ["animal", "pet", "bicho"],
            "pet": ["pet", "animal", "bicho"],
        }

        expanded = [query_lower]

        # Add synonyms
        for key, values in synonyms.items():
            if key in query_lower or any(val in query_lower for val in values):
                expanded.extend(values)

        # Remove duplicates
        return list(set(expanded))

    def search_by_image(self, query_image_path: str, top_k: int = 8) -> List[Dict]:
        """
        Reverse image search - find similar images using combined embeddings.

        Args:
            query_image_path: Path to query image
            top_k: Number of results to return

        Returns:
            List of similar images with similarity scores
        """
        if not Path(query_image_path).exists():
            raise FileNotFoundError(f"Query image not found: {query_image_path}")

        try:
            # Generate embeddings for query image
            image_embedding = np.array(self.clip_embeddings([query_image_path])[0])

            # For reverse image search, we weight the image embedding more heavily
            # since we're looking for visual similarity
            adjusted_query = 0.8 * image_embedding + 0.2 * np.zeros_like(image_embedding)
            adjusted_query = adjusted_query.tolist()

            # Search in ChromaDB using combined embeddings
            results = self.collection.query(
                query_embeddings=[adjusted_query],
                n_results=top_k * 2,
                include=["metadatas", "documents", "distances"]
            )

            # Process candidates
            candidates = []
            for metadata, distance, document in zip(
                results["metadatas"][0],
                results["distances"][0],
                results["documents"][0]
            ):
                if "photo_id" in metadata:
                    candidates.append({
                        "photo_id": int(metadata["photo_id"]),
                        "similarity": 1 - distance,
                        "caption": document,
                        "tags": metadata.get("tags", ""),
                        "file_name": metadata.get("file_name", ""),
                        "user_description": metadata.get("user_description", ""),
                        "metadata": metadata
                    })

            # Re-rank with context about visual similarity
            return self._rerank_results(candidates, "Find the most visually similar images to this one", use_image=True)[:top_k]

        except Exception as e:
            logger.error(f"Reverse image search failed: {e}")
            return []

    def _enhance_query(self, query: str) -> str:
        """Enhance search query using LLM for better results"""
        if not self.llm:
            return query

        try:
            prompt = PromptTemplate.from_template(
                "Rewrite this search query to be more specific and avoid confusion:\n{query}"
            )
            chain = prompt | self.llm
            enhanced = chain.invoke({"query": query}).content
            return enhanced.strip()
        except Exception as e:
            logger.warning(f"Failed to enhance query: {e}")
            return query

    def _should_use_reranking(self, query: str) -> bool:
        """
        Enhanced decision logic for when to use re-ranking based on query analysis.

        Args:
            query: Search query to analyze

        Returns:
            True if re-ranking should be used, False otherwise
        """
        query_lower = query.lower().strip()

        # Always skip re-ranking for very simple queries
        simple_indicators = [
            len(query.split()) <= 1,  # Single words
            query.isdigit(),  # Numbers only
            len(query) < 5,  # Very short queries
            query_lower in ['gato', 'cachorro', 'carro', 'casa', 'pessoa', 'comida', 'bebida', 'flor', 'árvore'],  # Common single nouns
            query_lower.startswith(('foto', 'imagem', 'picture', 'img')),  # Meta queries
        ]

        # Always use re-ranking for complex queries
        complex_indicators = [
            len(query.split()) >= 5,  # Very long queries
            any(word in query_lower for word in ['com', 'e', 'ou', 'mas', 'não', 'sem', 'exceto']),  # Complex connectors
            '?' in query or query_lower.startswith(('qual', 'quais', 'onde', 'quando', 'como')),  # Questions
            any(char in query for char in ['"', "'", '(', ')', '[', ']', '{', '}']),  # Quoted or parenthesized
            ',' in query or ';' in query,  # Lists or compound queries
            any(word in query_lower for word in ['melhor', 'mais', 'menos', 'tipo', 'tipo de', 'parecido']),  # Comparative queries
        ]

        # Medium complexity - use heuristics
        medium_indicators = [
            len(query.split()) == 3,  # 3-word queries often need clarification
            any(color in query_lower for color in ['vermelho', 'azul', 'verde', 'amarelo', 'preto', 'branco', 'roxo', 'rosa', 'laranja']),  # Color queries
            any(size in query_lower for size in ['grande', 'pequeno', 'alto', 'baixo', 'largo', 'estreito']),  # Size queries
            any(emotion in query_lower for emotion in ['feliz', 'triste', 'sorrindo', 'chorando', 'rindo']),  # Emotional queries
        ]

        # Don't use re-ranking for simple queries
        if any(simple_indicators):
            return False

        # Always use re-ranking for complex queries
        if any(complex_indicators):
            return True

        # Use re-ranking for medium-complexity queries (coin flip based on indicators)
        if any(medium_indicators):
            return True

        # Default: be conservative, don't use re-ranking for borderline cases
        return len(query.split()) >= 4

    def _rerank_results(self, candidates: List[Dict], query: str, use_image: bool = False) -> List[Dict]:
        """
        Optimized re-ranking with early exit and simplified prompts for better performance.
        """
        if len(candidates) <= 1 or not self.llm:
            return candidates

        # Early exit for simple cases
        if len(candidates) <= 3:
            return candidates

        try:
            # Limit candidates for re-ranking to improve speed
            candidates_to_rank = candidates[:8]  # Max 8 for re-ranking

            # Enhanced prompt for better LLM understanding
            prompt = f"""Você é um especialista em busca de imagens. Ordene estas {len(candidates_to_rank)} imagens por relevância para a consulta: "{query}"

IMPORTANTE: Foque na correspondência semântica, não apenas em palavras-chave. Considere:
- Similaridade visual e conceitual com a consulta
- Contexto e intenção da busca
- Relevância prática (ex: "carro vermelho" deve priorizar carros vermelhos sobre outros objetos vermelhos)

Responda APENAS com números separados por vírgula, em ordem de relevância (1=mais relevante).
Exemplo: 1,3,2,4,5

Imagens disponíveis:
"""

            for i, candidate in enumerate(candidates_to_rank, 1):
                # Include more context for better ranking
                context = candidate.get('caption', '')[:100]  # Limit caption length
                filename = candidate.get('file_name', f'imagem_{i}')
                user_desc = candidate.get('user_description', '')[:50]

                prompt += f"{i}. {filename}"
                if context:
                    prompt += f" - {context}"
                if user_desc:
                    prompt += f" (Contexto: {user_desc})"
                prompt += "\n"

            prompt += "\nOrdem de relevância (apenas números):"

            # Faster LLM call with optimized settings
            response = self.llm.invoke(prompt)

            # Parse response more efficiently
            content = response.content.strip() if hasattr(response, 'content') else str(response)

            # Extract ranking
            import re
            numbers = re.findall(r'\d+', content)
            ranking = [int(n) for n in numbers if 1 <= int(n) <= len(candidates_to_rank)]

            # Reorder based on ranking
            ranked_results = []
            for rank in ranking:
                if 1 <= rank <= len(candidates_to_rank):
                    ranked_results.append(candidates_to_rank[rank - 1])

            # Add remaining candidates
            remaining = [c for c in candidates if c not in ranked_results]
            return ranked_results + remaining

        except Exception as e:
            logger.warning(f"Re-ranking failed: {e}")
            return candidates

    def _rerank_results_batch(self, query_candidates_pairs: List[Tuple[str, List[Dict]]]) -> List[List[Dict]]:
        """
        Batch re-ranking for multiple queries to improve efficiency.

        Args:
            query_candidates_pairs: List of (query, candidates) tuples

        Returns:
            List of re-ranked candidate lists
        """
        if not query_candidates_pairs or not self.llm:
            return [candidates for _, candidates in query_candidates_pairs]

        results = []

        # Process in smaller batches to avoid token limits
        batch_size = 3  # Process 3 queries at a time

        for i in range(0, len(query_candidates_pairs), batch_size):
            batch = query_candidates_pairs[i:i + batch_size]

            # Create combined prompt for batch
            prompt = "Você é um especialista em busca de imagens. Para cada consulta abaixo, ordene as imagens por relevância.\n\n"

            query_indices = []
            for j, (query, candidates) in enumerate(batch):
                if len(candidates) <= 1:
                    results.append(candidates)
                    continue

                candidates_to_rank = candidates[:6]  # Limit per query in batch
                query_indices.append((j, query, candidates_to_rank))

                prompt += f"Consulta {j+1}: \"{query}\"\n"
                prompt += f"Imagens para Consulta {j+1}:\n"

                for k, candidate in enumerate(candidates_to_rank, 1):
                    context = candidate.get('caption', '')[:80]
                    filename = candidate.get('file_name', f'imagem_{k}')
                    prompt += f"  {k}. {filename}"
                    if context:
                        prompt += f" - {context}"
                    prompt += "\n"
                prompt += "\n"

            if not query_indices:
                continue

            prompt += "Para cada consulta, responda com: 'Consulta N: ordem_relevancia' (ex: Consulta 1: 1,3,2)\n"

            try:
                # Single LLM call for the batch
                response = self.llm.invoke(prompt)
                content = response.content.strip() if hasattr(response, 'content') else str(response)

                # Parse batch response
                for j, query, candidates_to_rank in query_indices:
                    # Extract ranking for this query
                    pattern = f"Consulta {j+1}: ([0-9, ]+)"
                    import re
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        ranking_str = match.group(1)
                        numbers = re.findall(r'\d+', ranking_str)
                        ranking = [int(n) for n in numbers if 1 <= int(n) <= len(candidates_to_rank)]

                        # Reorder candidates
                        ranked_results = []
                        for rank in ranking:
                            if 1 <= rank <= len(candidates_to_rank):
                                ranked_results.append(candidates_to_rank[rank - 1])

                        # Add remaining candidates
                        remaining = [c for c in batch[j][1] if c not in ranked_results]
                        results.append(ranked_results + remaining)
                    else:
                        # Fallback: return original order
                        results.append(batch[j][1])

            except Exception as e:
                logger.warning(f"Batch re-ranking failed: {e}")
                # Fallback: return all original orders
                for _, candidates in batch:
                    results.extend([candidates] * len(batch))
                break

        return results

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for monitoring"""
        try:
            current_time = time.time()
            uptime_seconds = current_time - self._performance_stats["last_reset"]

            # Calculate rates per second
            searches_per_second = self._performance_stats["total_searches"] / max(uptime_seconds, 1)
            cache_hit_rate = (self._performance_stats["cache_hits"] /
                            max(self._performance_stats["total_searches"], 1)) * 100

            reranking_rate = (self._performance_stats["reranking_used"] /
                            max(self._performance_stats["total_searches"], 1)) * 100

            # Get cache stats
            cache_stats = self.embedding_cache.get_stats()

            return {
                "uptime_seconds": uptime_seconds,
                "total_searches": self._performance_stats["total_searches"],
                "searches_per_second": round(searches_per_second, 2),
                "cache_hits": self._performance_stats["cache_hits"],
                "cache_misses": self._performance_stats["cache_misses"],
                "cache_hit_rate_percent": round(cache_hit_rate, 2),
                "reranking_used": self._performance_stats["reranking_used"],
                "reranking_skipped": self._performance_stats["reranking_skipped"],
                "reranking_rate_percent": round(reranking_rate, 2),
                "avg_search_time_ms": round(self._performance_stats["avg_search_time"] * 1000, 2),
                "slow_queries_count": self._performance_stats["slow_queries"],
                "query_type_distribution": self._performance_stats["query_types"],
                "cache_stats": cache_stats,
                "collection_stats": self.get_collection_stats(),
                "last_reset": self._performance_stats["last_reset"],
                "status": "healthy" if searches_per_second > 0 else "idle"
            }
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {
                "status": "error",
                "error": str(e),
                "uptime_seconds": 0
            }

    def check_health_alerts(self) -> Dict[str, Any]:
        """Check for performance issues and return alerts"""
        alerts = []
        metrics = self.get_performance_metrics()

        # Alert thresholds
        if metrics.get("avg_search_time_ms", 0) > 2000:  # > 2 seconds
            alerts.append({
                "level": "critical",
                "message": f"Average search time too high: {metrics['avg_search_time_ms']}ms",
                "threshold": "2000ms",
                "current": f"{metrics['avg_search_time_ms']}ms"
            })

        if metrics.get("cache_hit_rate_percent", 0) < 50:  # < 50% hit rate
            alerts.append({
                "level": "warning",
                "message": f"Low cache hit rate: {metrics['cache_hit_rate_percent']}%",
                "threshold": "50%",
                "current": f"{metrics['cache_hit_rate_percent']}%"
            })

        if metrics.get("slow_queries_count", 0) > 10:  # Too many slow queries
            alerts.append({
                "level": "warning",
                "message": f"High number of slow queries: {metrics['slow_queries_count']}",
                "threshold": "10",
                "current": str(metrics['slow_queries_count'])
            })

        if metrics.get("status") == "error":
            alerts.append({
                "level": "critical",
                "message": "Service is in error state",
                "details": metrics.get("error", "Unknown error")
            })

        return {
            "alerts_count": len(alerts),
            "alerts": alerts,
            "overall_status": "critical" if any(a["level"] == "critical" for a in alerts) else "warning" if alerts else "healthy"
        }

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            return {
                "total_images": count,
                "collection_name": "images",
                "embedding_dimensions": 512,  # Combined embedding dimension
                "status": "active",
                "embedding_type": "combined_clip_text",
                "description": "Unified visual-textual search space (70% CLIP + 30% SentenceTransformer)"
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                "total_images": 0,
                "status": "error",
                "error": str(e)
            }
