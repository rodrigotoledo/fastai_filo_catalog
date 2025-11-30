import os
import time
import logging
from sqlalchemy.orm import sessionmaker
from app.db.database import engine
from app.models.photo import Photo
from app.jobs.photo_processor import enqueue_photo_processing

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_unprocessed_photos():
    """
    Verifica e enfileira processamento para fotos não processadas
    """
    logger.info("Verificando fotos não processadas...")

    # Criar sessão do banco
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        # Buscar fotos não processadas
        unprocessed_photos = db.query(Photo).filter(Photo.processed == False).all()

        if not unprocessed_photos:
            logger.info("Nenhuma foto não processada encontrada")
            return

        logger.info(f"Encontradas {len(unprocessed_photos)} fotos não processadas")

        # Enfileirar processamento para cada foto
        for photo in unprocessed_photos:
            try:
                job_id = enqueue_photo_processing(photo.id)
                logger.info(f"Job enfileirado para foto {photo.id} (arquivo: {photo.original_filename})")
            except Exception as e:
                logger.error(f"Erro ao enfileirar job para foto {photo.id}: {str(e)}")

    except Exception as e:
        logger.error(f"Erro ao verificar fotos não processadas: {str(e)}")
    finally:
        db.close()

def run_scheduler(interval_seconds: int = 300):  # 5 minutos por padrão
    """
    Executa o scheduler periodicamente
    """
    logger.info(f"Iniciando scheduler com intervalo de {interval_seconds} segundos")

    while True:
        try:
            process_unprocessed_photos()
        except Exception as e:
            logger.error(f"Erro no scheduler: {str(e)}")

        logger.info(f"Aguardando {interval_seconds} segundos até próxima verificação...")
        time.sleep(interval_seconds)

if __name__ == "__main__":
    # Executar scheduler
    interval = int(os.getenv('SCHEDULER_INTERVAL_SECONDS', '300'))  # 5 minutos por padrão
    run_scheduler(interval)
