# Dockerfile para a aplicação FastAPI
FROM python:3.12-slim

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Criar usuário não-root com UID 1000 (mesmo do host)
RUN groupadd -r -g 1000 appuser && useradd -r -u 1000 -g appuser appuser

# Criar diretório da aplicação
WORKDIR /app

# Alterar proprietário do diretório
RUN chown appuser:appuser /app

# Copiar requirements e instalar dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY . .

# Criar diretório para uploads e ajustar permissões
RUN mkdir -p uploads && chown -R appuser:appuser /app

# Expor porta
EXPOSE 8000

# Ajustar permissões em tempo de execução (caso o volume sobrescreva)
USER root
RUN chown -R appuser:appuser /app/uploads 2>/dev/null || true
USER appuser

# Comando para iniciar a aplicação
CMD ["python", "run.py"]
