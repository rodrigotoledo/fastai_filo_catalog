from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import uuid
import logging
import time
from app.db.database import get_db
from app.services.photo_service import PhotoService
from app.models.photo import Photo
from app.schemas.photo import PhotoResponse, PaginatedPhotosResponse, SearchResponse, PhotoUploadRequest, SearchResultResponse

logger = logging.getLogger(__name__)

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

                # Check if agent processing is enabled
                use_agent = os.getenv("USE_LANGCHAIN_AGENTS", "false").lower() == "true"

                abs_file_path = os.path.join(os.getcwd(), photo.file_path)
                if os.path.exists(abs_file_path):
                    if use_agent:
                        # Use agent for intelligent processing
                        result_id = visual_search.add_image_with_agent(abs_file_path, photo.id, description)
                        print(f"Agent processed image {file.filename}: {result_id}")
                    else:
                        # Use direct processing
                        visual_search.add_image(abs_file_path, photo.id, description)
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
    count: int = Form(1, description="Número de imagens a baixar (1-10)", ge=1, le=10),
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

# ============================================================================
# MONITORING ENDPOINTS
# ============================================================================

@router.get("/metrics")
async def get_performance_metrics():
    """
    Get comprehensive performance metrics for monitoring and alerting
    """
    try:
        from app.services.visual_search_service import VisualSearchService

        visual_search = VisualSearchService()
        metrics = visual_search.get_performance_metrics()

        return {
            "service": "photo_search",
            "version": "4.0",  # Phase 4: Monitoring
            "timestamp": time.time(),
            "metrics": metrics
        }

    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics collection failed: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Comprehensive health check with performance alerts
    """
    try:
        from app.services.visual_search_service import VisualSearchService

        visual_search = VisualSearchService()
        metrics = visual_search.get_performance_metrics()
        alerts = visual_search.check_health_alerts()

        # Overall health status
        status_code = 200
        if alerts["overall_status"] == "critical":
            status_code = 503  # Service Unavailable
        elif alerts["overall_status"] == "warning":
            status_code = 200  # OK but with warnings

        response = {
            "service": "photo_search",
            "status": alerts["overall_status"],
            "timestamp": time.time(),
            "version": "4.0",
            "uptime_seconds": metrics.get("uptime_seconds", 0),
            "alerts": alerts,
            "quick_stats": {
                "total_searches": metrics.get("total_searches", 0),
                "avg_response_time_ms": metrics.get("avg_search_time_ms", 0),
                "cache_hit_rate": metrics.get("cache_hit_rate_percent", 0),
                "total_images": metrics.get("collection_stats", {}).get("total_images", 0)
            }
        }

        return response

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

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

@router.get("/search/smart", response_model=SearchResponse)
def search_photos_with_agent(
    q: str = Query(..., description="Search query text"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(12, ge=1, le=50, description="Number of results per page"),
    db: Session = Depends(get_db)
):
    """
    Search photos using LangChain agent for intelligent query processing
    """
    logger.info(f"🔍 SMART SEARCH ENDPOINT CALLED with query: {q}")
    try:
        from app.services.langchain_agents import ImageProcessingAgent
        from app.services.ai_service import AIService
        from app.services.visual_search_service import VisualSearchService

        photo_service = PhotoService(db)
        visual_search = VisualSearchService()
        ai_service = AIService()

        if not ai_service.llm:
            logger.warning("❌ LLM not available, falling back to regular search")
            return search_photos_by_text(q, page, page_size, db)

        # Create agent for intelligent search
        logger.info(f"🤖 Creating ImageProcessingAgent...")
        agent = ImageProcessingAgent(ai_service.llm, visual_search, ai_service, photo_service)
        logger.info(f"🤖 Agent created successfully")

        # Agent analyzes query and decides search strategy
        search_task = f"""
        Você é um especialista em busca de imagens. Analise esta consulta e encontre as imagens mais relevantes: "{q}"

        SUA MISSÃO:
        1. Entenda a intenção da consulta (o que o usuário está procurando?)
        2. Expanda a consulta com sinônimos se necessário (português/inglês)
        3. Escolha a melhor estratégia de busca:
           - "semantic": busca semântica padrão
           - "expanded": expandir com sinônimos para mais resultados
        4. Use a ferramenta smart_search para executar a busca
        5. Analise os resultados e refine se necessário

        SEJA INTELIGENTE SOBRE:
        - Termos em português vs inglês
        - Múltiplos significados de palavras
        - Contexto e intenção
        - Relevância visual vs textual

        Execute a busca usando a ferramenta smart_search e retorne os melhores resultados.
        """

        logger.info(f"🤖 About to execute agent...")
        try:
            # Agent decides and executes the search
            logger.info(f"🤖 Executing agent search for query: {q}")
            agent_result = agent.agent_executor.invoke({"input": search_task})
            logger.info(f"🤖 Agent result: {agent_result}")

            # Extract search results from agent response
            # For now, fallback to direct search while we parse agent results
            all_results = visual_search.search_by_text(q, top_k=1000)

        except Exception as agent_error:
            logger.warning(f"Agent search failed, using direct search: {agent_error}")
            logger.warning(f"⚠️ Agent failed: {agent_error}")
            # Fallback to direct search
            all_results = visual_search.search_by_text(q, top_k=1000)

        # Paginate results
        total_results = len(all_results)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_results = all_results[start_idx:end_idx]

        # Fetch Photo objects
        search_results = []
        for result in paginated_results:
            photo = photo_service.get_photo(result["photo_id"])
            if photo:
                search_results.append(SearchResultResponse(
                    photo=PhotoResponse.from_orm(photo),
                    similarity_score=result["similarity"]
                ))

        return SearchResponse(results=search_results)

    except ImportError:
        # Fallback if agents not available
        return search_photos_by_text(q, page, page_size, db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent search failed: {str(e)}")

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

@router.post("/process-with-agent")
async def process_image_with_agent(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None, description="Descrição opcional da imagem"),
    db: Session = Depends(get_db)
):
    """
    Process image using LangChain agent for intelligent analysis
    """
    try:
        from app.services.langchain_agents import ImageProcessingAgent
        from app.services.ai_service import AIService
        from app.services.visual_search_service import VisualSearchService

        # Save file temporarily
        temp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Initialize services
        ai_service = AIService()
        visual_search = VisualSearchService()

        if not ai_service.llm:
            raise HTTPException(status_code=500, detail="LLM not available for agent processing")

        # Create and use agent
        agent = ImageProcessingAgent(ai_service.llm, visual_search, ai_service)
        result = agent.process_image(temp_path, description)

        # Clean up temp file
        os.remove(temp_path)

        return {
            "success": result["success"],
            "agent_response": result.get("agent_response", ""),
            "processing_details": result
        }

    except ImportError:
        raise HTTPException(status_code=500, detail="LangChain agents not available")
    except Exception as e:
        # Clean up on error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Agent processing failed: {str(e)}")

@router.get("/search/fast", response_model=SearchResponse)
def search_photos_fast(
    q: str = Query(..., description="Search query text"),
    top_k: int = Query(12, ge=1, le=50, description="Number of results"),
    db: Session = Depends(get_db)
):
    """
    Fast semantic search using optimized ChromaDB with caching (Phase 4)
    Bypasses LangChain agent for maximum speed
    """
    try:
        from app.services.visual_search_service import VisualSearchService

        visual_search = VisualSearchService()
        photo_service = PhotoService(db)

        # Use optimized search_by_text method directly
        search_results = visual_search.search_by_text(q, top_k=top_k)

        # Convert to API response format
        results = []
        for result in search_results:
            photo = photo_service.get_photo(result['photo_id'])
            if photo:
                results.append(SearchResultResponse(
                    photo=PhotoResponse.from_orm(photo),
                    similarity_score=result['similarity']
                ))

        return SearchResponse(results=results)

    except Exception as e:
        logger.error(f"Fast search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# ============================================================================
# SEARCH ENDPOINTS
# ============================================================================

from app.services.visual_search_service import VisualSearchService


