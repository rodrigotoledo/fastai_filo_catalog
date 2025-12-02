import os
import shutil
from pathlib import Path
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session
from app.models.photo import Photo
import aiofiles
from app.jobs.photo_processor import enqueue_photo_processing
from app.services.ai_service import AIService
from typing import List, Tuple
import httpx
import uuid

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

class PhotoService:
    def __init__(self, db: Session):
        self.db = db

    async def save_photo(self, file: UploadFile, user_description: str = None) -> Photo:
        print(f"DEBUG: Starting save_photo for {file.filename}")  # Debug
        # Validar tipo de arquivo
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Only image files are allowed")

        # Gerar nome único para o arquivo
        file_extension = Path(file.filename).suffix
        import uuid
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = UPLOAD_DIR / unique_filename

        # Salvar arquivo no disco
        try:
            content = await file.read()
            print(f"DEBUG: Content length: {len(content)}")  # Debug
            with open(file_path, "wb") as buffer:
                buffer.write(content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

        # Reset file pointer for potential reuse
        await file.seek(0)

        # Criar registro no banco
        photo = Photo(
            filename=unique_filename,
            original_filename=file.filename,
            file_path=str(file_path),
            file_size=len(content),  # Usar len(content) em vez de stat
            content_type=file.content_type,
            user_description=user_description
        )

        self.db.add(photo)
        self.db.commit()
        self.db.refresh(photo)

        # Enfileirar processamento de IA
        try:
            job_id = enqueue_photo_processing(photo.id)
            print(f"DEBUG: Job enfileirado: {job_id}")
        except Exception as e:
            print(f"WARNING: Não foi possível enfileirar processamento: {str(e)}")

        return photo

    async def populate_photo(self, term: str, count: int = 1) -> List[Photo]:
        """
        Baixa múltiplas imagens do LoremFlickr e salva como fotos
        """
        if count < 1 or count > 10:
            raise HTTPException(status_code=400, detail="Count must be between 1 and 10")

        photos = []
        for i in range(count):
            print(f"DEBUG: Starting populate_photo {i+1}/{count} for term: {term}")

            # Validar e sanitizar o termo
            if not term or not term.strip():
                raise HTTPException(status_code=400, detail="Term cannot be empty")

            # Limpar e codificar o termo para URL
            import urllib.parse
            clean_term = term.strip()

            # Lista de termos bloqueados ou problemáticos
            blocked_terms = ['nude', 'naked', 'sex', 'porn', 'adult']
            if any(blocked.lower() in clean_term.lower() for blocked in blocked_terms):
                raise HTTPException(status_code=400, detail="Term contains inappropriate content")

            encoded_term = urllib.parse.quote(clean_term)

            # URL do LoremFlickr com parâmetro rand para evitar cache
            import time
            rand_param = int(time.time() * 1000) + i  # timestamp + índice para variar
            url = f"https://loremflickr.com/800/600/{encoded_term}?rand={rand_param}"

            print(f"DEBUG: URL gerada: {url}")

            # Fazer download da imagem
            async with httpx.AsyncClient(follow_redirects=True) as client:
                try:
                    response = await client.get(url)
                    response.raise_for_status()

                    # Verificar se a resposta é realmente uma imagem
                    content_type = response.headers.get('content-type', '')
                    if not content_type.startswith('image/'):
                        print(f"DEBUG: Invalid response type: {content_type}, trying fallback")
                        # Tentar novamente sem termo específico (imagem aleatória)
                        fallback_url = f"https://loremflickr.com/800/600?rand={rand_param}"
                        response = await client.get(fallback_url)
                        response.raise_for_status()

                        content_type = response.headers.get('content-type', '')
                        if not content_type.startswith('image/'):
                            raise HTTPException(status_code=500, detail=f"Invalid fallback response type: {content_type}")

                        content = response.content
                        if len(content) < 1000:
                            raise HTTPException(status_code=500, detail="Downloaded fallback content is too small")

                        # Ajustar o termo para indicar que foi fallback
                        actual_term = f"{clean_term} (random)"
                    else:
                        content = response.content
                        actual_term = clean_term

                        # Verificar se o conteúdo não está vazio
                        if len(content) < 1000:  # Imagens devem ter pelo menos 1KB
                            raise HTTPException(status_code=500, detail="Downloaded content is too small")

                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 403:
                        # Tentar novamente sem termo específico (imagem aleatória)
                        print(f"DEBUG: Term '{clean_term}' blocked, trying random image")
                        fallback_url = f"https://loremflickr.com/800/600?rand={rand_param}"
                        try:
                            response = await client.get(fallback_url)
                            response.raise_for_status()

                            content_type = response.headers.get('content-type', '')
                            if not content_type.startswith('image/'):
                                raise HTTPException(status_code=500, detail=f"Invalid fallback response type: {content_type}")

                            content = response.content
                            if len(content) < 1000:
                                raise HTTPException(status_code=500, detail="Downloaded fallback content is too small")

                            # Ajustar o termo para indicar que foi fallback
                            actual_term = f"{clean_term} (random)"

                        except Exception as fallback_e:
                            raise HTTPException(status_code=500, detail=f"Failed to download fallback image: {str(fallback_e)}")
                    else:
                        raise HTTPException(status_code=500, detail=f"Failed to download image: {str(e)}")
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to download image: {str(e)}")

            # Gerar nome único para o arquivo
            unique_filename = f"{uuid.uuid4()}.jpg"
            file_path = UPLOAD_DIR / unique_filename

            # Salvar arquivo no disco
            try:
                with open(file_path, "wb") as buffer:
                    buffer.write(content)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

            # Criar registro no banco
            photo = Photo(
                filename=unique_filename,
                original_filename=f"{actual_term}_{i+1}.jpg",
                file_path=str(file_path),
                file_size=len(content),
                content_type="image/jpeg",
                user_description=actual_term
            )

            self.db.add(photo)
            self.db.commit()
            self.db.refresh(photo)

            # Enfileirar processamento de IA
            try:
                job_id = enqueue_photo_processing(photo.id)
                print(f"DEBUG: Job enfileirado para foto populada {i+1}: {job_id}")
            except Exception as e:
                print(f"WARNING: Não foi possível enfileirar processamento: {str(e)}")

            photos.append(photo)

        # Ordenar por data de upload descendente (mais recentes primeiro)
        photos_sorted = sorted(photos, key=lambda p: p.uploaded_at, reverse=True)
        return photos_sorted

    def get_photos(self, page: int = 1, page_size: int = 12):
        """
        Get photos with pagination
        """
        if page < 1:
            page = 1

        skip = (page - 1) * page_size
        query = self.db.query(Photo)
        total = query.count()
        photos = query.order_by(Photo.uploaded_at.desc()).offset(skip).limit(page_size).all()

        total_pages = (total + page_size - 1) // page_size  # Ceiling division

        return {
            "photos": photos,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }

    def get_photo(self, photo_id: int):
        return self.db.query(Photo).filter(Photo.id == photo_id).first()

    def search_similar_photos(self, query_text: str = None, photo_id: int = None, limit: int = 10):
        """
        Busca fotos similares por texto ou por outra foto
        """
        ai_service = AIService()

        # Buscar todas as fotos processadas
        processed_photos = self.db.query(Photo).filter(
            Photo.processed == True,
            Photo.embedding.isnot(None)
        ).all()

        if not processed_photos:
            return {"results": [], "message": "Nenhuma foto processada encontrada"}

        # Preparar embeddings
        embeddings = [(photo.id, photo.embedding) for photo in processed_photos]

        # Buscar por texto ou por foto similar
        if query_text:
            # Buscar mais resultados para ter chance de aplicar boost
            similar_photos = ai_service.find_similar_by_text(query_text, embeddings, limit * 10)
        elif photo_id:
            # Buscar foto de referência
            ref_photo = self.db.query(Photo).filter(Photo.id == photo_id).first()
            if not ref_photo or not ref_photo.embedding:
                raise HTTPException(status_code=404, detail="Foto de referência não encontrada ou não processada")

            similar_photos = ai_service.search_similar_images(ref_photo.embedding, embeddings, limit)
        else:
            raise HTTPException(status_code=400, detail="Deve fornecer query_text ou photo_id")

        # Buscar fotos completas pelos IDs
        photo_ids = [photo_id for photo_id, _ in similar_photos]
        photos = self.db.query(Photo).filter(Photo.id.in_(photo_ids)).order_by(Photo.uploaded_at.desc()).all()

        # Criar mapa de ID -> foto
        photo_map = {photo.id: photo for photo in photos}

        # Montar resultado com scores
        results = []
        for photo_id, score in similar_photos:
            if photo_id in photo_map:
                photo = photo_map[photo_id]
                boosted_score = score

                # Boost para correspondência exata de texto
                if query_text and photo.user_description:
                    query_lower = query_text.lower()
                    desc_lower = photo.user_description.lower()

                    # Boost se o texto de busca estiver contido na descrição
                    if query_lower in desc_lower:
                        boosted_score = min(1.0, score + 0.3)  # Adicionar 0.3 de boost, máximo 1.0
                    # Boost menor se palavras individuais coincidirem
                    elif any(word in desc_lower for word in query_lower.split()):
                        boosted_score = min(1.0, score + 0.1)  # Adicionar 0.1 de boost

                results.append({
                    "photo": photo,
                    "similarity_score": boosted_score
                })

        # Reordenar por score após boost
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        results = results[:limit]

        return {"results": results}

    def get_processing_stats(self):
        """
        Retorna estatísticas de processamento
        """
        total_photos = self.db.query(Photo).count()
        processed_photos = self.db.query(Photo).filter(Photo.processed == True).count()
        unprocessed_photos = total_photos - processed_photos

        return {
            "total_photos": total_photos,
            "processed_photos": processed_photos,
            "unprocessed_photos": unprocessed_photos,
            "processing_percentage": round((processed_photos / total_photos * 100) if total_photos > 0 else 0, 2)
        }
