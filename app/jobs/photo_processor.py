import os
import redis
from rq import Queue, Connection
from rq.job import Job
from sqlalchemy.orm import sessionmaker
from app.db.database import engine
from app.models.photo import Photo
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Conectar ao Redis
redis_conn = redis.from_url(os.getenv('REDIS_URL', 'redis://redis:6379'))

def process_photo_job(photo_id: int):
    """
    Job para processar uma foto com IA
    """
    logger.info(f"Processando foto ID: {photo_id}")

    # Importar aqui para evitar dependências circulares
    from app.services.ai_service import AIService

    # Criar sessão do banco
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        # Buscar foto
        photo = db.query(Photo).filter(Photo.id == photo_id).first()
        if not photo:
            logger.error(f"Foto {photo_id} não encontrada")
            return

        if photo.processed:
            logger.info(f"Foto {photo_id} já foi processada")
            return

        # Verificar se arquivo existe
        if not os.path.exists(photo.file_path):
            logger.error(f"Arquivo não encontrado: {photo.file_path}")
            return

        # Processar com IA
        ai_service = AIService()
        embedding, description = ai_service.process_image(photo.file_path, photo.user_description)

        # Atualizar foto no banco
        photo.embedding = embedding
        photo.description = description
        photo.processed = True

        db.commit()
        logger.info(f"Foto {photo_id} processada com sucesso")

        # Adicionar ao índice de busca visual
        try:
            from app.services.visual_search_service import VisualSearchService
            visual_search = VisualSearchService()
            visual_search.add_image(photo.file_path, photo.id, photo.user_description)
            logger.info(f"Foto {photo_id} adicionada ao índice de busca")
        except Exception as e:
            logger.error(f"Erro ao adicionar foto {photo_id} ao índice de busca: {str(e)}")
            # Não falhar o processamento se o índice falhar

    except Exception as e:
        logger.error(f"Erro ao processar foto {photo_id}: {str(e)}")
        db.rollback()
    finally:
        db.close()

def enqueue_photo_processing(photo_id: int):
    """
    Adiciona job de processamento na fila
    """
    with Connection(redis_conn):
        q = Queue('photo_processing')
        job = q.enqueue(process_photo_job, photo_id)
        logger.info(f"Job enfileirado para foto {photo_id}: {job.id}")
        return job.id

def get_processing_status(job_id: str):
    """
    Verifica status de um job
    """
    with Connection(redis_conn):
        try:
            job = Job.fetch(job_id)
            return {
                'id': job.id,
                'status': job.get_status(),
                'result': job.result,
                'error': str(job.exc_info) if job.exc_info else None
            }
        except Exception as e:
            return {'error': str(e)}
