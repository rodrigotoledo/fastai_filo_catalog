#!/usr/bin/env python3
"""
Script para popular embeddings das fotos existentes no banco de dados.
Rode uma vez só após adicionar a coluna embedding.
"""

import os
import sys
from pathlib import Path

# Adicionar o diretório raiz ao path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from app.db.database import SessionLocal
from app.models.photo import Photo
from app.services.ai_service import AIService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def populate_embeddings():
    """Popula embeddings para todas as fotos que ainda não têm"""
    db = SessionLocal()
    try:
        ai_service = AIService()

        # Buscar fotos sem embedding
        photos_without_embedding = db.query(Photo).filter(Photo.embedding.is_(None)).all()

        if not photos_without_embedding:
            logger.info("Todas as fotos já têm embeddings!")
            return

        logger.info(f"Encontradas {len(photos_without_embedding)} fotos sem embedding")

        for i, photo in enumerate(photos_without_embedding, 1):
            try:
                logger.info(f"Processando {i}/{len(photos_without_embedding)}: {photo.original_filename}")

                # Verificar se o arquivo existe
                if not os.path.exists(photo.file_path):
                    logger.warning(f"Arquivo não encontrado: {photo.file_path}")
                    continue

                # Gerar embedding do texto (descrição)
                full_text = f"{photo.user_description or ''} {photo.description or ''}".strip()
                if full_text:
                    embedding = ai_service.generate_clip_text_embedding(full_text)
                    if embedding and not all(x == 0.0 for x in embedding):
                        photo.embedding = embedding
                        logger.info(f"Embedding gerado para {photo.original_filename}")
                    else:
                        logger.warning(f"Falha ao gerar embedding para {photo.original_filename}")
                else:
                    logger.warning(f"Sem texto para gerar embedding: {photo.original_filename}")

                # Commit a cada foto para não perder progresso
                db.commit()

            except Exception as e:
                logger.error(f"Erro ao processar {photo.original_filename}: {e}")
                db.rollback()
                continue

        logger.info("Processamento concluído!")

    except Exception as e:
        logger.error(f"Erro geral: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    populate_embeddings()
