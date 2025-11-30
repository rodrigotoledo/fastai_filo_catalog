import os
import shutil
from pathlib import Path
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session
from app.models.photo import Photo
import aiofiles

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
