import os
import time
import logging
from sqlalchemy.orm import sessionmaker
from app.db.database import engine
from app.models.photo import Photo
from app.models.client import Client
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

def process_unprocessed_clients():
    """
    Processa clientes não processados com IA (execução síncrona)
    """
    logger.info("Verificando clientes não processados...")

    # Criar sessão do banco
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        # Buscar clientes não processados
        unprocessed_clients = db.query(Client).filter(Client.processed == False).all()

        if not unprocessed_clients:
            logger.info("Nenhum cliente não processado encontrado")
            return

        logger.info(f"Encontrados {len(unprocessed_clients)} clientes não processados")

        # Importar ClientService aqui para evitar import circular
        from app.services.client_service import ClientService

        # Processar cada cliente
        for client in unprocessed_clients:
            try:
                client_service = ClientService(db)
                processed_client = client_service.process_client(client.id)
                logger.info(f"Cliente {client.id} ({client.name}) processado com IA")
            except Exception as e:
                logger.error(f"Erro ao processar cliente {client.id}: {str(e)}")

    except Exception as e:
        logger.error(f"Erro ao verificar clientes não processados: {str(e)}")
    finally:
        db.close()

def run_scheduler(interval_seconds: int = 10):
    """
    Executa o scheduler periodicamente
    """
    logger.info(f"Iniciando scheduler com intervalo de {interval_seconds} segundos")

    while True:
        try:
            process_unprocessed_photos()
            process_unprocessed_clients()
        except Exception as e:
            logger.error(f"Erro no scheduler: {str(e)}")

        logger.info(f"Aguardando {interval_seconds} segundos até próxima verificação...")
        time.sleep(interval_seconds)

if __name__ == "__main__":
    # Executar scheduler
    interval = int(os.getenv('SCHEDULER_INTERVAL_SECONDS', '10'))  # 5 minutos por padrão
    run_scheduler(interval)
