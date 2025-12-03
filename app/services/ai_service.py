import numpy as np
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import logging

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        # Usar CLIP para processamento real de imagens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Usando dispositivo: {self.device}")

        try:
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model.to(self.device)
            self.embedding_dim = 512  # CLIP ViT-B/32 tem 512 dimensões
            logger.info("CLIP model carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar CLIP model: {e}")
            # Fallback para embeddings simples
            self.model = None
            self.processor = None
            self.embedding_dim = 768

    def process_image(self, image_path: str, user_prompt: str = None) -> Tuple[List[float], str]:
        """
        Processa uma imagem usando CLIP e retorna embedding + descrição
        """
        try:
            if self.model is None:
                # Fallback para método simples
                filename = Path(image_path).name
                embedding = self._simple_embedding()
                description = f"Imagem processada (fallback): {filename}"
                if user_prompt:
                    description += f" | Contexto: {user_prompt}"
                return embedding, description

            # Carregar e processar imagem
            image = Image.open(image_path).convert('RGB')

            # Se temos prompt do usuário, combinar com processamento da imagem
            if user_prompt and user_prompt.strip():
                # Estratégia: Processar imagem + texto juntos
                combined_text = f"Esta é uma foto de {user_prompt}"

                # Processar imagem
                image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                image_features = self.model.get_image_features(**image_inputs)

                # Processar texto combinado
                text_inputs = self.processor(text=[combined_text], return_tensors="pt", padding=True).to(self.device)
                text_features = self.model.get_text_features(**text_inputs)

                # Estratégia adaptativa: se o user_prompt contém palavras-chave específicas,
                # dar mais peso ao texto para melhorar matching exato
                keywords = ['gato', 'cachorro', 'pássaro', 'ave', 'carro', 'praia', 'montanha']
                has_specific_keywords = any(keyword in user_prompt.lower() for keyword in keywords)

                if has_specific_keywords:
                    # Dar muito mais peso ao texto para matching exato
                    combined_embedding = 0.1 * image_features + 0.9 * text_features
                else:
                    # Peso balanceado para descrições gerais
                    combined_embedding = 0.5 * image_features + 0.5 * text_features

                # Normalizar
                combined_embedding = combined_embedding / combined_embedding.norm(dim=-1, keepdim=True)

                embedding = combined_embedding.detach().cpu().numpy().flatten().tolist()
                description = f"Imagem processada com contexto: {user_prompt}"

            else:
                # Processamento normal da imagem
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)

                embedding = image_features.detach().cpu().numpy().flatten().tolist()
                filename = Path(image_path).name
                description = f"Imagem processada com IA: {filename}"

            # Garantir que embedding seja lista de floats
            embedding = [float(x) for x in embedding]

            logger.info(f"Imagem {Path(image_path).name} processada com sucesso")
            return embedding, description

        except Exception as e:
            logger.error(f"Erro ao processar imagem {image_path}: {str(e)}")
            # Fallback
            filename = Path(image_path).name
            return self._simple_embedding(), f"Erro no processamento: {filename}"

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
        Converte texto para embedding usando CLIP
        """
        try:
            if self.model is None:
                # Fallback para método simples
                return self._simple_embedding()

            # Processar texto com CLIP
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)

            # Normalizar e converter para lista
            embedding = text_features.detach().cpu().numpy().flatten().tolist()
            embedding = [float(x) for x in embedding]

            logger.info(f"Texto processado: '{text}'")
            return embedding

        except Exception as e:
            logger.error(f"Erro ao processar texto '{text}': {str(e)}")
            return self._simple_embedding()

    def find_similar_by_text(self, query_text: str, embeddings: List[Tuple[int, List[float]]], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Busca imagens por texto
        """
        query_embedding = self.text_to_embedding(query_text)
        return self.search_similar_images(query_embedding, embeddings, top_k)

    def process_client_text(self, client_data: dict, user_description: str = None) -> Tuple[List[float], str]:
        """
        Processa dados de cliente e gera embedding + descrição usando CLIP
        """
        try:
            # Criar texto descritivo conciso do cliente (limitar tamanho para evitar truncamento)
            text_parts = []

            # Informações principais (sempre incluir)
            if client_data.get('name'):
                text_parts.append(client_data['name'])

            # Adicionar apenas cidade/estado do primeiro endereço (mais útil para busca)
            addresses = client_data.get('addresses', [])
            if addresses and addresses[0].get('city'):
                addr = addresses[0]
                location = f"{addr.get('city', '')}, {addr.get('state', '')}".strip(", ")
                if location:
                    text_parts.append(location)

            # Adicionar nickname se existir (útil para busca informal)
            if client_data.get('nickname'):
                text_parts.append(client_data['nickname'])

            # Combinar com descrição do usuário se fornecida
            base_text = " ".join(text_parts)

            if user_description and user_description.strip():
                # Limitar descrição do usuário para não exceder limite de tokens
                user_desc_short = user_description[:100]  # Limitar a 100 caracteres
                combined_text = f"{base_text} {user_desc_short}".strip()
            else:
                combined_text = base_text

            # Garantir que não está vazio
            if not combined_text.strip():
                combined_text = "Cliente sem informações"

            # Limitar tamanho total para evitar problemas com CLIP
            if len(combined_text) > 200:
                combined_text = combined_text[:200] + "..."

            # Gerar embedding usando CLIP
            embedding = self.text_to_embedding(combined_text)

            # Criar descrição da IA
            ai_description = f"Cliente processado com IA. Dados: {base_text}"
            if user_description:
                ai_description += f" | Contexto adicional: {user_description}"

            logger.info(f"Cliente '{client_data.get('name', 'Unknown')}' processado com sucesso")
            return embedding, ai_description

        except Exception as e:
            logger.error(f"Erro ao processar cliente {client_data.get('name', 'Unknown')}: {str(e)}")
            # Fallback
            return self._simple_embedding(), f"Erro no processamento do cliente: {client_data.get('name', 'Unknown')}"

    def find_similar_clients(self, query_text: str, embeddings: List[Tuple[int, List[float]]], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Busca clientes similares por texto
        """
        query_embedding = self.text_to_embedding(query_text)
        return self.search_similar_images(query_embedding, embeddings, top_k)
