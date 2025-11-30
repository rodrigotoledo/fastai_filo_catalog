import numpy as np
from pathlib import Path
from typing import List, Tuple
import hashlib

class AIService:
    def __init__(self):
        # Por enquanto, vamos usar embeddings simples baseados em hash
        # TODO: Implementar modelos reais de IA quando necessário
        self.embedding_dim = 768

    def process_image(self, image_path: str) -> Tuple[List[float], str]:
        """
        Processa uma imagem e retorna embedding + descrição
        """
        try:
            # Criar embedding baseado no nome do arquivo (determinístico)
            filename = Path(image_path).name
            embedding = self._filename_to_embedding(filename)

            description = f"Imagem processada: {filename}"

            return embedding, description

        except Exception as e:
            print(f"Erro ao processar imagem {image_path}: {str(e)}")
            # Fallback
            return self._simple_embedding(), f"Imagem processada: {Path(image_path).name}"

    def _filename_to_embedding(self, filename: str) -> List[float]:
        """Cria embedding determinístico baseado no nome do arquivo"""
        hash_obj = hashlib.md5(filename.encode())
        seed = int(hash_obj.hexdigest(), 16) % (2**32)
        np.random.seed(seed)
        return np.random.normal(0, 1, self.embedding_dim).tolist()

    def _simple_embedding(self) -> List[float]:
        """Embedding simples para fallback"""
        np.random.seed(42)  # Para consistência
        return np.random.normal(0, 1, self.embedding_dim).tolist()

    def search_similar_images(self, query_embedding: List[float], embeddings: List[Tuple[int, List[float]]], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Busca imagens similares baseado no embedding
        """
        query_vec = np.array(query_embedding)

        similarities = []
        for photo_id, embedding in embeddings:
            if embedding is not None and len(embedding) > 0:
                try:
                    emb_vec = np.array(embedding)
                    # Similaridade cosseno
                    norm_product = np.linalg.norm(query_vec) * np.linalg.norm(emb_vec)
                    if norm_product > 0:
                        similarity = np.dot(query_vec, emb_vec) / norm_product
                        similarities.append((photo_id, float(similarity)))
                except:
                    continue

        # Ordenar por similaridade (maior primeiro)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def text_to_embedding(self, text: str) -> List[float]:
        """
        Converte texto para embedding
        """
        # Embedding baseado no texto (determinístico)
        hash_obj = hashlib.md5(text.encode())
        seed = int(hash_obj.hexdigest(), 16) % (2**32)
        np.random.seed(seed)
        return np.random.normal(0, 1, self.embedding_dim).tolist()

    def find_similar_by_text(self, query_text: str, embeddings: List[Tuple[int, List[float]]], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Busca imagens por texto
        """
        query_embedding = self.text_to_embedding(query_text)
        return self.search_similar_images(query_embedding, embeddings, top_k)
