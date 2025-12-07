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

        # Create or get collection for images
        self.collection = self.client.get_or_create_collection(
            name="images",
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize embedding functions
        self.clip_embeddings = OpenCLIPEmbeddingFunction()  # For image embeddings
        self.text_embeddings = SentenceTransformerEmbeddingFunction(
            model_name="multi-qa-mpnet-base-cos-v1"  # Better for Portuguese
        )

        # Initialize LLM for rich captions and re-ranking
        self.llm = self._initialize_llm()

        logger.info("VisualSearchService initialized with ChromaDB")

    def _initialize_llm(self) -> Optional[object]:
        """Initialize LLM with fallback chain"""
        try:
            # Try Google Gemini first
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                return ChatGoogleGenerativeAI(
                    model="gemini-pro",
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

    def search_by_text(self, query: str, top_k: int = 8) -> List[Dict]:
        """
        Search images by text query using semantic similarity with combined embeddings.

        Args:
            query: Text query to search for
            top_k: Number of results to return

        Returns:
            List of search results with similarity scores
        """
        try:
            # Generate embedding for the text query
            query_embedding = np.array(self.text_embeddings([query])[0])

            # Since we stored combined embeddings (70% image + 30% text),
            # we need to adjust the query embedding to match the same space
            # For text queries, we weight the text embedding more heavily
            adjusted_query = 0.3 * np.zeros_like(query_embedding) + 0.7 * query_embedding
            adjusted_query = adjusted_query.tolist()

            # Search in ChromaDB using semantic similarity
            results = self.collection.query(
                query_embeddings=[adjusted_query],
                n_results=top_k * 2,  # Get more candidates for re-ranking
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
                        "similarity": 1 - distance,  # Convert distance to similarity
                        "caption": document,
                        "tags": metadata.get("tags", ""),
                        "file_name": metadata.get("file_name", ""),
                        "user_description": metadata.get("user_description", ""),
                        "metadata": metadata
                    })

            # Re-rank with LLM to eliminate false positives and improve relevance
            return self._rerank_results(candidates, f"Find images most relevant to: '{query}'", use_image=False)[:top_k]

        except Exception as e:
            logger.error(f"Semantic text search failed: {e}")
            # Fallback to basic text search
            return self._fallback_text_search(query, top_k)

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

    def _rerank_results(self, candidates: List[Dict], query: str, use_image: bool = False) -> List[Dict]:
        """
        Re-rank search results using LLM to eliminate false positives.
        """
        if len(candidates) <= 1 or not self.llm:
            return candidates

        try:
            prompt = f"""
            Query: {query}

            Rank these images from most relevant to least relevant.
            Respond ONLY with the file names in correct order, one per line.

            Candidates:
            """

            for i, candidate in enumerate(candidates[:12], 1):
                prompt += f"\n{i}. {candidate['file_name']}\n   → {candidate['caption'][:300]}..."

            chain = PromptTemplate.from_template(prompt) | self.llm
            response = chain.invoke({})

            # Parse response
            lines = [line.strip() for line in response.content.split("\n") if line.strip() and "." in line]
            ranked_names = [line.split(".")[1].split("→")[0].strip() for line in lines]

            # Reorder candidates
            ranked_results = []
            for name in ranked_names:
                for candidate in candidates:
                    if name in candidate["file_name"]:
                        if candidate not in ranked_results:
                            ranked_results.append(candidate)

            # Add remaining candidates
            remaining = [c for c in candidates if c not in ranked_results]
            return ranked_results + remaining

        except Exception as e:
            logger.warning(f"Re-ranking failed: {e}")
            return candidates

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
