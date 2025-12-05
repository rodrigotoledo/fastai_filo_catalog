from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import os
from app.db.database import get_db
from app.services.photo_service import PhotoService
from app.models.photo import Photo
from app.schemas.photo import PhotoResponse, PaginatedPhotosResponse, SearchResponse, PhotoUploadRequest

router = APIRouter()

@router.post("/upload", response_model=List[PhotoResponse])
async def upload_photos(
    files: List[UploadFile] = File(...),
    description: Optional[str] = Form(None, description="Descrição opcional para todas as fotos"),
    db: Session = Depends(get_db)
):
    """
    Upload multiple photos with optional description
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    photo_service = PhotoService(db)
    uploaded_photos = []

    for file in files:
        try:
            photo = await photo_service.save_photo(file, user_description=description)
            uploaded_photos.append(photo)
        except Exception as e:
            # Se um arquivo falhar, continua com os outros
            # Ou pode decidir falhar tudo dependendo da lógica
            print(f"Error uploading file {file.filename}: {str(e)}")
            continue

    return uploaded_photos

@router.post("/populate", response_model=List[PhotoResponse])
async def populate_photo(
    term: str = Form(..., description="Termo para buscar imagem no LoremFlickr"),
    count: int = Form(4, description="Número de imagens a baixar (1-10)", ge=1, le=10),
    db: Session = Depends(get_db)
):
    """
    Baixa múltiplas imagens do LoremFlickr e adiciona ao banco com o termo como descrição
    """
    photo_service = PhotoService(db)
    photos = await photo_service.populate_photo(term, count)
    return photos

@router.get("/", response_model=PaginatedPhotosResponse)
def get_photos(
    page: int = Query(1, ge=1, description="Page number (starting from 1)"),
    page_size: int = Query(12, ge=1, le=100, description="Number of photos per page"),
    db: Session = Depends(get_db)
):
    """
    Get photos with pagination
    """
    photo_service = PhotoService(db)
    return photo_service.get_photos(page=page, page_size=page_size)

@router.get("/file/{photo_id}")
def get_photo_file(photo_id: int, db: Session = Depends(get_db)):
    """
    Get the actual photo file by ID
    """
    photo_service = PhotoService(db)
    photo = photo_service.get_photo(photo_id)
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")

    file_path = photo.file_path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, media_type=photo.content_type)

@router.get("/{photo_id}", response_model=PhotoResponse)
def get_photo(photo_id: int, db: Session = Depends(get_db)):
    """
    Get a specific photo by ID
    """
    photo_service = PhotoService(db)
    photo = photo_service.get_photo(photo_id)
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")
    return photo

@router.get("/search/text", response_model=SearchResponse)
def search_photos_by_text(
    q: str = Query(..., description="Search query text"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results"),
    db: Session = Depends(get_db)
):
    """
    Search photos by text similarity using AI
    """
    photo_service = PhotoService(db)
    return photo_service.search_similar_photos_by_text(query_text=q, limit=limit)

@router.get("/processing/stats")
def get_processing_stats(db: Session = Depends(get_db)):
    """
    Get processing statistics
    """
    photo_service = PhotoService(db)
    return photo_service.get_processing_stats()
