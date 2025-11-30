#!/usr/bin/env python3
"""
Worker para processar jobs de IA das fotos
"""
import os
import sys
from pathlib import Path

# Adicionar diretório raiz ao path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from rq import Worker, Queue, Connection
from app.jobs.photo_processor import redis_conn

# Conectar às filas
with Connection(redis_conn):
    worker = Worker(['photo_processing'])
    worker.work()
