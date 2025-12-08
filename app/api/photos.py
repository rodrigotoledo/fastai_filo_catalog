# app/routers/photos.py
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List
import os
import uuid
import logging

from app.db.database import get_db
from app.services.photo_service import PhotoService
from app.services.ai_service import AIService
from app.models.photo import Photo
from app.schemas.photo import PhotoResponse, SearchResponse, SearchResultResponse, PaginatedPhotosResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["photos"])

ai_service = AIService()  # singleton – já carrega o CLIP uma vez só


# ============================
# UPLOAD + embedding automático
# ============================
@router.post("/upload", response_model=List[PhotoResponse])
async def upload_photos(
    files: List[UploadFile] = File(...),
    description: str | None = None,
    db: Session = Depends(get_db),
):
    if not files:
        raise HTTPException(400, "Nenhum arquivo enviado")

    photo_service = PhotoService(db)
    uploaded = []

    for file in files:
        # 1. salva no disco + cria registro no banco
        photo = await photo_service.save_photo(file, user_description=description)

        try:
            # 2. gera embedding CLIP direto dos bytes (sem tocar no disco de novo)
            content = await file.read()
            embedding = ai_service.generate_clip_image_embedding(content)

            photo.image_embedding = embedding
            db.commit()
            logger.info(f"Embedding gerado e salvo para foto {photo.id}")

        except Exception as e:
            logger.warning(f"Falha ao gerar embedding da foto {photo.id}: {e}")
            # não quebra o upload se o embedding falhar

        uploaded.append(photo)

    return uploaded


# ============================
# BUSCA POR TEXTO (o que você vai usar 99% do tempo)
# ============================
@router.get("/search", response_model=PaginatedPhotosResponse)
def search(
    q: str = Query(..., description="O que você quer encontrar? Ex: vestido floral vermelho"),
    limit: int = Query(12, ge=1, le=50),
    db: Session = Depends(get_db),
):
    query_vec = ai_service.generate_clip_text_embedding(q)
    if not query_vec:
        raise HTTPException(500, "Erro ao gerar embedding do texto")

    sql = text("""
        SELECT id, original_filename, user_description, file_path,
               image_embedding <=> :vec AS distance
        FROM photos
        WHERE image_embedding IS NOT NULL
        ORDER BY distance
        LIMIT :limit
    """)

    rows = db.execute(
        sql,
        {"vec": f"[{','.join(map(str, query_vec))}]", "limit": limit}
    ).fetchall()

    results = []
    for r in rows:
        confidence = round((1 - r.distance) * 100, 1)
        if confidence >= 18:  # filtro de relevância
            photo_service = PhotoService(db)
            photo = photo_service.get_photo(r.id)
            if photo:
                photo_response = PhotoResponse.from_orm(photo)
                photo_response.similarity_score = confidence
                results.append(photo_response)

    return PaginatedPhotosResponse(
        results=results,
        total=len(results),
        page=1,
        page_size=limit,
        total_found=len(results),  # para buscas, sempre 1 página
        has_next=False,  # buscas não têm paginação
        has_prev=False   # buscas não têm paginação
    )


# ============================
# BUSCA POR IMAGEM (reverse search)
# ============================
@router.post("/search/image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile = File(...),
    limit: int = Query(12, ge=1, le=50),
    db: Session = Depends(get_db),
):
    content = await file.read()
    query_vec = ai_service.generate_clip_image_embedding(content)
    if not query_vec:
        raise HTTPException(500, "Erro ao gerar embedding da imagem enviada")

    # mesma query do /search, só muda a origem do vetor
    sql = text("""
        SELECT id, original_filename, user_description,
               image_embedding <=> :vec AS distance
        FROM photos
        WHERE image_embedding IS NOT NULL
        ORDER BY distance
        LIMIT :limit
    """)

    rows = db.execute(sql, {"vec": f"[{','.join(map(str, query_vec))}]", "limit": limit}).fetchall()

    results = []
    for r in rows:
        confidence = round((1 - r.distance) * 100, 1)
        photo = PhotoService(db).get_photo(r.id)
        results.append(
            SearchResultResponse(
                photo=PhotoResponse.from_orm(photo),
                similarity_score=confidence,
            )
        )

    return SearchResponse(results=results)


# ============================
# Lista paginada + download da foto
# ============================
@router.get("/", response_model=PaginatedPhotosResponse)
def list_photos(page: int = Query(1, ge=1), page_size: int = Query(12, ge=1, le=100), db: Session = Depends(get_db)):
    return PhotoService(db).get_photos(page=page, page_size=page_size)


@router.get("/file/{photo_id}")
def get_file(photo_id: int, db: Session = Depends(get_db)):
    photo = PhotoService(db).get_photo(photo_id)
    if not photo or not os.path.exists(photo.file_path):
        raise HTTPException(404, "Foto não encontrada")
    return FileResponse(photo.file_path, media_type=photo.content_type)


# ============================
# Migração das fotos antigas (roda UMA vez)
# ============================
@router.post("/migrate-embeddings")
def migrate_old_photos(db: Session = Depends(get_db)):
    photos = db.query(Photo).filter(Photo.image_embedding.is_(None)).all()
    count = 0
    for p in photos:
        try:
            with open(p.file_path, "rb") as f:
                emb = ai_service.generate_clip_image_embedding(f.read())
            p.image_embedding = emb
            db.commit()
            count += 1
        except Exception as e:
            logger.error(f"Erro na foto {p.id}: {e}")
    return {"migrated": count, "total_pending": len(photos)}
