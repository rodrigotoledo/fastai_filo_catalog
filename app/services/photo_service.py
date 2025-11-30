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

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

class PhotoService:
    def __init__(self, db: Session):
        self.db = db

    async def save_photo(self, file: UploadFile) -> Photo:
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
            content_type=file.content_type
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

    def get_photos(self, page: int = 1, page_size: int = 12):
        """
        Get photos with pagination
        """
        if page < 1:
            page = 1

        skip = (page - 1) * page_size
        query = self.db.query(Photo)
        total = query.count()
        photos = query.offset(skip).limit(page_size).all()

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
            similar_photos = ai_service.find_similar_by_text(query_text, embeddings, limit)
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
        photos = self.db.query(Photo).filter(Photo.id.in_(photo_ids)).all()

        # Criar mapa de ID -> foto
        photo_map = {photo.id: photo for photo in photos}

        # Montar resultado com scores
        results = []
        for photo_id, score in similar_photos:
            if photo_id in photo_map:
                results.append({
                    "photo": photo_map[photo_id],
                    "similarity_score": score
                })

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
