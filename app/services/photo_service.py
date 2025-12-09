import os
import shutil
from pathlib import Path
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session
from app.models.photo import Photo
import aiofiles
from app.services.ai_service import AIService
from typing import List, Dict
import httpx
import uuid
import logging

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

class PhotoService:
    def __init__(self, db: Session):
        self.db = db

    async def save_photo(self, file: UploadFile, user_description: str = None) -> Photo:
        """
        Salva foto no banco + upload para Gemini File Search Store
        """
        # Validar tipo de arquivo
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Only image files are allowed")

        # Gerar nome único para o arquivo
        file_extension = Path(file.filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = UPLOAD_DIR / unique_filename

        # Salvar arquivo no disco
        try:
            content = await file.read()
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
            file_size=len(content),
            content_type=file.content_type,
            image_data=content,  # Save image bytes for CLIP processing
            user_description=user_description or "Foto sem descrição"
        )

        self.db.add(photo)
        self.db.commit()
        self.db.refresh(photo)

        # Upload para Gemini File Search Store (se disponível)
        try:
            # Upload done. Now Process with AI (Gemini)
            ai_service = AIService()

            # 1. Generate Rich Description using Gemini Multimodal
            logger.info(f"Generating rich description for {unique_filename}...")
            rich_description = ai_service._generate_description_gemini(str(file_path), user_description)

            # Update photo with description
            photo.description = rich_description

            # 2. Generate Embedding for the description using CLIP (OpenAI)
            # We combine user_description + rich_description for better semantic coverage
            full_text_context = f"{user_description or ''} {rich_description}"
            logger.info(f"Generating CLIP text embedding for {unique_filename}...")
            embedding = ai_service.generate_clip_text_embedding(full_text_context)

            # 3. Generate Image Embedding using OpenAI CLIP
            logger.info(f"Generating CLIP image embedding for {unique_filename}...")
            image_embedding = ai_service.generate_clip_image_embedding(content)

            if embedding:
                photo.embedding = embedding
            if image_embedding:
                photo.image_embedding = image_embedding
                photo.processed = True

            self.db.add(photo)
            self.db.commit()
            self.db.refresh(photo)

        except Exception as e:
            logger.error(f"Failed to process AI fields for {unique_filename}: {e}")
            # Non-blocking, we still saved the photo

        return photo

    async def populate_photo(self, term: str, count: int = 1) -> List[Photo]:
        """
        Baixa múltiplas imagens do LoremFlickr e salva como fotos
        """
        if count < 1 or count > 10:
            raise HTTPException(status_code=400, detail="Count must be between 1 and 10")

        photos = []
        for i in range(count):

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

            # Tentar LoremFlickr primeiro
            try:
                encoded_term = urllib.parse.quote(clean_term)
                url = f"https://image.pollinations.ai/prompt/{encoded_term}"

                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.get(url)
                    response.raise_for_status()

                    content_type = response.headers.get('content-type', '')
                    if not content_type.startswith('image/'):
                        raise HTTPException(status_code=500, detail=f"Invalid response type: {content_type}")

                    content = response.content
                    if len(content) < 1000:
                        raise HTTPException(status_code=500, detail="Downloaded content is too small")

                    actual_term = clean_term

            except Exception as e:
                # Fallback inteligente: tentar termos mais genéricos
                fallback_terms = []

                # Quebrar o termo em palavras e tentar combinações
                words = clean_term.split()
                if len(words) > 1:
                    # Tentar primeira palavra
                    fallback_terms.append(words[0])
                    # Tentar segunda palavra
                    if len(words) > 1:
                        fallback_terms.append(words[1])

                # Termos genéricos relacionados
                if any(word in clean_term.lower() for word in ['cachorro', 'dog', 'cão']):
                    fallback_terms.extend(['dog', 'animal', 'pet'])
                elif any(word in clean_term.lower() for word in ['gato', 'cat']):
                    fallback_terms.extend(['cat', 'animal', 'pet'])
                elif any(word in clean_term.lower() for word in ['praia', 'beach', 'mar']):
                    fallback_terms.extend(['beach', 'sea', 'ocean', 'nature'])
                elif any(word in clean_term.lower() for word in ['cidade', 'city', 'urbano']):
                    fallback_terms.extend(['city', 'urban', 'street'])

                # Adicionar termos genéricos sempre disponíveis
                fallback_terms.extend(['nature', 'landscape', 'house', 'building'])

                # Tentar cada termo de fallback
                success = False
                for fallback_term in fallback_terms[:5]:  # Limitar a 5 tentativas
                    try:
                        encoded_fallback = urllib.parse.quote(fallback_term)
                        url = f"https://image.pollinations.ai/prompt/{encoded_fallback}"

                        async with httpx.AsyncClient(timeout=30) as client:
                            response = await client.get(url)
                            response.raise_for_status()

                            content_type = response.headers.get('content-type', '')
                            if not content_type.startswith('image/'):
                                continue

                            content = response.content
                            if len(content) < 1000:
                                continue

                            actual_term = f"{clean_term} (fallback: {fallback_term})"
                            success = True
                            break

                    except Exception as fallback_e:
                        continue

                # Se nenhum fallback funcionou, usar imagem aleatória mas marcar claramente
                if not success:
                    try:
                        random_url = f"https://image.pollinations.ai/prompt/random"

                        async with httpx.AsyncClient(timeout=30) as client:
                            response = await client.get(random_url)
                            response.raise_for_status()

                            content_type = response.headers.get('content-type', '')
                            if not content_type.startswith('image/'):
                                raise HTTPException(status_code=500, detail=f"Invalid random response type: {content_type}")

                            content = response.content
                            if len(content) < 1000:
                                raise HTTPException(status_code=500, detail="Downloaded random content is too small")

                            actual_term = f"{clean_term} (random image - no relevant images found)"

                    except Exception as random_e:
                        raise HTTPException(status_code=500, detail=f"Failed to download random image: {str(random_e)}")
                fallback_terms = []

                # Quebrar o termo em palavras e tentar combinações
                words = clean_term.split()
                if len(words) > 1:
                    # Tentar primeira palavra
                    fallback_terms.append(words[0])
                    # Tentar segunda palavra
                    if len(words) > 1:
                        fallback_terms.append(words[1])

                # Termos genéricos relacionados
                if any(word in clean_term.lower() for word in ['cachorro', 'dog', 'cão']):
                    fallback_terms.extend(['dog', 'animal', 'pet'])
                elif any(word in clean_term.lower() for word in ['gato', 'cat']):
                    fallback_terms.extend(['cat', 'animal', 'pet'])
                elif any(word in clean_term.lower() for word in ['praia', 'beach', 'mar']):
                    fallback_terms.extend(['beach', 'sea', 'ocean', 'nature'])
                elif any(word in clean_term.lower() for word in ['cidade', 'city', 'urbano']):
                    fallback_terms.extend(['city', 'urban', 'street'])

                # Adicionar termos genéricos
                fallback_terms.extend(['nature', 'landscape', 'random'])

                # Tentar cada termo de fallback
                success = False
                for fallback_term in fallback_terms[:3]:  # Limitar a 3 tentativas
                    try:
                        encoded_fallback = urllib.parse.quote(fallback_term)
                        url = f"https://image.pollinations.ai/prompt/{encoded_fallback}"

                        async with httpx.AsyncClient(timeout=30) as client:
                            response = await client.get(url)
                            response.raise_for_status()

                            content_type = response.headers.get('content-type', '')
                            if not content_type.startswith('image/'):
                                continue

                            content = response.content
                            if len(content) < 1000:
                                continue

                            actual_term = f"{clean_term} (fallback: {fallback_term})"
                            success = True
                            break

                    except Exception as fallback_e:
                        continue

                # Se nenhum fallback funcionou, usar imagem aleatória mas marcar claramente
                if not success:
                    try:
                        random_url = f"https://image.pollinations.ai/prompt/random"

                        async with httpx.AsyncClient(timeout=30) as client:
                            response = await client.get(random_url)
                            response.raise_for_status()

                            content_type = response.headers.get('content-type', '')
                            if not content_type.startswith('image/'):
                                raise HTTPException(status_code=500, detail=f"Invalid random response type: {content_type}")

                            content = response.content
                            if len(content) < 1000:
                                raise HTTPException(status_code=500, detail="Downloaded random content is too small")

                            actual_term = f"{clean_term} (random image - no relevant images found)"

                    except Exception as random_e:
                        raise HTTPException(status_code=500, detail=f"Failed to download random image: {str(random_e)}")

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
                user_description=actual_term,
                description=actual_term
            )

            self.db.add(photo)
            self.db.commit()
            self.db.refresh(photo)

            photos.append(photo)

        return photos

    def get_photo(self, photo_id: int) -> Photo:
        """
        Busca uma foto específica por ID
        """
        return self.db.query(Photo).filter(Photo.id == photo_id).first()

    def get_processing_stats(self):
        """
        Retorna estatísticas detalhadas de processamento
        """
        from datetime import datetime, timedelta

        total_photos = self.db.query(Photo).count()
        processed_photos = self.db.query(Photo).filter(Photo.processed == True).count()
        unprocessed_photos = total_photos - processed_photos

        # Calcular porcentagem
        processing_percentage = round((processed_photos / total_photos * 100) if total_photos > 0 else 0, 2)

        # Pegar informações das últimas fotos processadas
        recent_processed = self.db.query(Photo).filter(Photo.processed == True)\
            .order_by(Photo.uploaded_at.desc())\
            .limit(5)\
            .all()

        recent_photos = []
        for photo in recent_processed:
            recent_photos.append({
                "id": photo.id,
                "filename": photo.filename,
                "uploaded_at": photo.uploaded_at.isoformat() if photo.uploaded_at else None,
                "has_description": photo.description is not None and len(photo.description.strip()) > 0,
                "has_embedding": photo.embedding is not None and len(photo.embedding) > 0
            })

        # Estatísticas de tempo (aproximado)
        avg_processing_time_per_photo = 15  # segundos estimados por foto
        estimated_remaining_time = unprocessed_photos * avg_processing_time_per_photo

        # Status do processamento
        status = "idle"
        if unprocessed_photos > 0:
            status = "processing"
        elif processed_photos == total_photos and total_photos > 0:
            status = "completed"

        return {
            "status": status,
            "total_photos": total_photos,
            "processed_photos": processed_photos,
            "unprocessed_photos": unprocessed_photos,
            "processing_percentage": processing_percentage,
            "estimated_remaining_seconds": estimated_remaining_time,
            "estimated_remaining_time": str(timedelta(seconds=estimated_remaining_time)),
            "recent_processed_photos": recent_photos,
            "last_updated": datetime.now().isoformat()
        }

    async def populate_photo(self, term: str, count: int = 1) -> List[Photo]:
        """
        Baixa múltiplas imagens de várias fontes e salva como fotos
        """
        if count < 1 or count > 10:
            raise HTTPException(status_code=400, detail="Count must be between 1 and 10")

        photos = []
        for i in range(count):

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

            # Lista de fontes de imagem com fallbacks
            import time
            rand_param = int(time.time() * 1000) + i  # timestamp + índice para variar

            image_sources = [
                # Pollinations.ai com termo específico
                f"https://image.pollinations.ai/prompt/{encoded_term}?rand={rand_param}",
                # Pollinations.ai com termo genérico
                f"https://image.pollinations.ai/prompt/{encoded_term.replace(' ', '%20')}?rand={rand_param}",
                # Picsum.photos com termo (fallback)
                f"https://picsum.photos/800/600?random={rand_param}",
                # LoremFlickr com termo
                f"https://loremflickr.com/800/600/{encoded_term}?lock={rand_param}",
                # LoremFlickr genérico
                f"https://loremflickr.com/800/600/nature?lock={rand_param}",
            ]

            content = None
            actual_term = clean_term
            download_success = False

            # Tentar cada fonte até conseguir
            for source_url in image_sources:
                try:
                    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
                        response = await client.get(source_url)
                        response.raise_for_status()

                        # Verificar se a resposta é realmente uma imagem
                        content_type = response.headers.get('content-type', '')
                        if content_type.startswith('image/'):
                            content = response.content

                            # Verificar se o conteúdo não está vazio
                            if len(content) >= 1000:  # Imagens devem ter pelo menos 1KB
                                download_success = True
                                print(f"Successfully downloaded from: {source_url}")
                                break
                            else:
                                print(f"Content too small from: {source_url}")
                        else:
                            print(f"Invalid content type from {source_url}: {content_type}")

                except Exception as e:
                    print(f"Failed to download from {source_url}: {str(e)}")
                    continue

            # Se nenhuma fonte funcionou, usar uma imagem de placeholder
            if not download_success:
                try:
                    # Último recurso: gerar uma imagem simples via data URL
                    import base64
                    # Criar uma imagem simples 1x1 pixel transparente como placeholder
                    placeholder_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                    content = base64.b64decode(placeholder_data)
                    actual_term = f"{clean_term} (placeholder)"
                    download_success = True
                    print("Using placeholder image as last resort")
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"All image sources failed, even placeholder: {str(e)}")

            if not download_success:
                raise HTTPException(status_code=500, detail="Failed to download image from any source")

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

            photos.append(photo)

        # Ordenar por data de upload descendente (mais recentes primeiro)
        photos_sorted = sorted(photos, key=lambda p: p.uploaded_at, reverse=True)
        return photos_sorted

    def get_photo(self, photo_id: int):
        return self.db.query(Photo).filter(Photo.id == photo_id).first()

    def get_photo(self, photo_id: int) -> Photo:
        """
        Busca uma foto específica por ID
        """
        return self.db.query(Photo).filter(Photo.id == photo_id).first()

    def get_photos(self, page: int = 1, page_size: int = 12, processed_only: bool = False) -> Dict:
        """
        Busca fotos com paginação. Opcionalmente filtra apenas fotos processadas.
        """
        offset = (page - 1) * page_size

        # Base query
        query = self.db.query(Photo)

        # Aplicar filtro se solicitado
        if processed_only:
            query = query.filter(Photo.processed == True)

        photos_query = query.all()
        total_photos = query.count()
        total_found = total_photos

        # Converter objetos Photo para dicionários serializáveis
        photos_data = []
        for photo in photos_query:
            photos_data.append({
                "id": photo.id,
                "filename": photo.filename,
                "original_filename": photo.original_filename,
                "file_path": photo.file_path,
                "file_size": photo.file_size,
                "content_type": photo.content_type,
                "uploaded_at": photo.uploaded_at.isoformat() if photo.uploaded_at else None,
                "processed": photo.processed,
                "description": photo.description,
                "user_description": photo.user_description
            })

        return {
            "results": photos_data,
            "total": total_photos,
            "page": page,
            "page_size": total_photos,
            "total_found": total_photos,
            "has_next": False,
            "has_prev": page > 1
        }
