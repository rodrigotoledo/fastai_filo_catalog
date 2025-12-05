from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import uuid
from app.db.database import get_db
from app.services.photo_service import PhotoService
from app.models.photo import Photo
from app.schemas.photo import PhotoResponse, PaginatedPhotosResponse, SearchResponse, PhotoUploadRequest, SearchResultResponse

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

            # Add to visual search index
            try:
                visual_search = VisualSearchService()
                # Use the photo data directly
                abs_file_path = os.path.join(os.getcwd(), photo.file_path)
                if os.path.exists(abs_file_path):
                    visual_search.add_image(abs_file_path, photo.id, user_description)
                else:
                    print(f"Warning: File not found for visual search: {abs_file_path}")
            except Exception as e:
                print(f"Warning: Failed to add {file.filename} to visual search: {str(e)}")
                # Don't fail the upload if visual search fails

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
    processed_only: bool = Query(False, description="Return only processed photos (with descriptions and embeddings)"),
    db: Session = Depends(get_db)
):
    """
    Get photos with pagination. Optionally filter to only processed photos.
    """
    photo_service = PhotoService(db)
    return photo_service.get_photos(page=page, page_size=page_size, processed_only=processed_only)

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
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(12, ge=1, le=50, description="Number of results per page"),
    db: Session = Depends(get_db)
):
    """
    Search photos by text using ChromaDB semantic search
    """
    try:
        photo_service = PhotoService(db)
        visual_search = VisualSearchService()
        # Get all results first (since ChromaDB doesn't support pagination directly)
        all_results = visual_search.search_by_text(q, top_k=1000)  # Get many results

        # Paginate the results
        total_results = len(all_results)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_results = all_results[start_idx:end_idx]

        # Fetch Photo objects for the results
        search_results = []
        for result in paginated_results:
            photo = photo_service.get_photo(result["photo_id"])
            if photo:
                search_results.append(SearchResultResponse(
                    photo=PhotoResponse.from_orm(photo),
                    similarity_score=result["similarity"]
                ))

        return SearchResponse(results=search_results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/search/image", response_model=SearchResponse)
async def search_photos_by_image(
    file: UploadFile = File(...),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(12, ge=1, le=50, description="Number of results per page"),
    db: Session = Depends(get_db)
):
    """
    Reverse image search - find visually similar photos
    """
    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        photo_service = PhotoService(db)
        visual_search = VisualSearchService()
        # Get all results first
        all_results = visual_search.search_by_image(temp_path, top_k=1000)

        # Clean up temp file
        os.unlink(temp_path)

        # Paginate the results
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_results = all_results[start_idx:end_idx]

        # Fetch Photo objects for the results
        search_results = []
        for result in paginated_results:
            photo = photo_service.get_photo(result["photo_id"])
            if photo:
                search_results.append(SearchResultResponse(
                    photo=PhotoResponse.from_orm(photo),
                    similarity_score=result["similarity"]
                ))

        return SearchResponse(results=search_results)

    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Reverse image search failed: {str(e)}")

@router.delete("/clear")
def clear_visual_search_index():
    """
    Clear all images from the visual search index
    """
    try:
        visual_search = VisualSearchService()
        # Delete and recreate collection
        collection_name = visual_search.collection.name
        visual_search.client.delete_collection(collection_name)
        visual_search.collection = visual_search.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        return {"message": "Visual search index cleared"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear index: {str(e)}")

@router.get("/debug/index")
def debug_visual_index():
    """
    Debug endpoint to check visual search index status
    """
    try:
        visual_search = VisualSearchService()
        count = visual_search.collection.count()
        return {
            "indexed_images": count,
            "cwd": os.getcwd(),
            "upload_dir_exists": os.path.exists("uploads"),
            "upload_dir_contents": os.listdir("uploads")[:5] if os.path.exists("uploads") else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")

@router.post("/reindex")
def reindex_all_photos(db: Session = Depends(get_db)):
    """
    Reindex all photos in the visual search database
    """
    try:
        photo_service = PhotoService(db)
        visual_search = VisualSearchService()

        # Clear existing index
        visual_search.collection.delete(where={})

        # Get all photos
        result = photo_service.get_photos(page=1, page_size=1000)
        photos = result['photos']

        indexed = 0
        for photo in photos:
            try:
                abs_file_path = os.path.join(os.getcwd(), photo['file_path'])
                if os.path.exists(abs_file_path):
                    visual_search.add_image(abs_file_path, photo['id'], photo['user_description'] or '')
                    indexed += 1
                else:
                    print(f"File not found: {abs_file_path}")
            except Exception as e:
                print(f"Error indexing {photo['id']}: {e}")

        return {
            "message": f"Reindexed {indexed} out of {len(photos)} photos",
            "total_photos": len(photos),
            "indexed": indexed
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reindex failed: {str(e)}")

@router.post("/reprocess")
def reprocess_all_photos(db: Session = Depends(get_db)):
    """
    Mark all photos as unprocessed to trigger reprocessing with new AI system
    """
    try:
        # Update all photos to mark as unprocessed
        updated_count = db.query(Photo).filter(Photo.processed == True).update({"processed": False})
        db.commit()

        return {
            "message": f"Marcadas {updated_count} fotos para reprocessamento",
            "details": "O scheduler irá reprocessar automaticamente as fotos com o novo sistema LangChain + OCR"
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Erro ao marcar fotos para reprocessamento: {str(e)}")

@router.get("/processing/stats")
def get_processing_stats(db: Session = Depends(get_db)):
    """
    Get processing statistics
    """
    photo_service = PhotoService(db)
    return photo_service.get_processing_stats()

# ============================================================================
# SEARCH ENDPOINTS
# ============================================================================

from app.services.visual_search_service import VisualSearchService


