#!/bin/bash
# Script para executar comandos Alembic no container da aplicação

if [ $# -eq 0 ]; then
    echo "Uso: $0 <comando alembic>"
    echo "Exemplo: $0 upgrade head"
    echo "         $0 revision --autogenerate -m 'mensagem'"
    exit 1
fi

docker compose exec --user appuser app alembic "$@"
