# Photo Finder Backend

Um sistema avançado de processamento e busca de imagens usando IA, construído com FastAPI, PostgreSQL com pgvector, e integração com múltiplos provedores de IA (OpenAI, Anthropic, Gemini, Local).

## 🚀 Visão Geral

O Photo Finder é uma aplicação backend que permite:

- **Upload e armazenamento** de imagens
- **Processamento automático** com IA (descrições + embeddings)
- **Busca semântica** por similaridade de texto
- **OCR para documentos** com extração de texto
- **Reprocessamento em lote** de imagens existentes

## 🏗️ Arquitetura

### Tecnologias Principais

- **Backend**: FastAPI (Python 3.12)
- **Banco**: PostgreSQL + pgvector (embeddings)
- **Fila**: Redis + RQ (processamento assíncrono)
- **IA**: LangChain com múltiplos provedores
- **OCR**: pytesseract + OpenCV
- **Containerização**: Docker Compose

### Componentes

```text
├── app/                    # Código da aplicação
│   ├── api/               # Endpoints FastAPI
│   ├── models/            # SQLModel (SQLAlchemy + Pydantic)
│   ├── services/          # Lógica de negócio
│   ├── db/                # Conexão e configuração do banco
│   └── jobs/              # Processamento assíncrono (RQ)
├── uploads/               # Arquivos de imagem
├── cache/                 # Modelos de IA em cache
├── alembic/               # Migrações do banco
└── docker-compose.yml     # Orquestração de containers
```

## ✨ Funcionalidades

### 📤 Upload de Imagens

- Upload múltiplo via API REST
- Validação de tipos (JPEG, PNG)
- Armazenamento otimizado
- Metadados automáticos

### 🤖 Processamento com IA

- **Descrições automáticas**: Geração de texto detalhado sobre o conteúdo da imagem
- **Embeddings semânticos**: Vetores de 512 dimensões para busca por similaridade
- **OCR integrado**: Extração de texto de documentos/imagens
- **Processamento assíncrono**: Background jobs com RQ

### 🔍 Busca Inteligente

- **Busca por texto**: Similaridade semântica (não palavras-chave exatas)
- **Resultados ranqueados**: Por relevância usando embeddings
- **Filtro opcional**: Apenas imagens processadas
- **Paginação**: Resultados eficientes

### 🖼️ Busca Visual Avançada (ChromaDB)

- **Busca semântica por texto**: Usando SentenceTransformers + re-ranking com LLM
- **Busca reversa por imagem**: Encontre imagens visualmente similares
- **Captions ricos com IA**: Descrições detalhadas geradas por multimodal LLMs
- **Embeddings duplos**: CLIP para imagens + SentenceTransformers para texto
- **Re-ranking inteligente**: LLM filtra falsos positivos

### 🔄 Reprocessamento

- **Endpoint dedicado**: Marcar todas as imagens para reprocessamento
- **Sistema de fallback**: OpenAI → Local → Anthropic → Gemini
- **Monitoramento**: Status em tempo real do progresso
- **Continuação automática**: Scheduler processa em background

## 🛠️ Instalação e Setup

### Pré-requisitos

- Docker e Docker Compose
- 4GB+ RAM (para modelos de IA)
- Chaves de API (opcional, mas recomendado)

### 1. Clone e Setup

```bash
git clone <repository>
cd photo-finder/backend
```

### 2. Configuração de Ambiente

```bash
# Copie o arquivo de exemplo
cp .env.example .env

# Configure as chaves de API (recomendado)
echo "OPENAI_API_KEY=sk-your-key" >> .env
echo "ANTHROPIC_API_KEY=sk-ant-your-key" >> .env
echo "GOOGLE_API_KEY=your-gemini-key" >> .env
```

### 3. Inicialização

```bash
# Build e start dos serviços
docker compose up -d

# Aplicar migrações do banco
docker compose exec app alembic upgrade head

# Verificar status
docker compose ps
```

### 4. Verificar Funcionamento

```bash
# API deve estar rodando em http://localhost:8000
curl http://localhost:8000/docs
```

## 📚 API Endpoints

### Upload de Imagens

```http
POST /api/v1/photos/upload
Content-Type: multipart/form-data

files: <arquivos de imagem>
description: "Descrição opcional"
```

### Listar Imagens

```http
GET /api/v1/photos/?page=1&page_size=12&processed_only=true
```

**Parâmetros:**

- `page`: Página atual (padrão: 1)
- `page_size`: Itens por página (padrão: 12, máx: 100)
- `processed_only`: Apenas imagens processadas (padrão: false)

### Busca por Texto

```http
GET /api/v1/photos/search/text?q=gato%20preto&limit=10
```

### Download de Imagem

```http
GET /api/v1/photos/file/{photo_id}
```

### Reprocessamento

```http
POST /api/v1/photos/reprocess
```

### Estatísticas de Processamento

```http
GET /api/v1/photos/processing/stats
```

**Resposta:**

```json
{
  "status": "processing",
  "total_photos": 76,
  "processed_photos": 23,
  "processing_percentage": 30.26,
  "estimated_remaining_time": "0:12:30",
  "recent_processed_photos": [...]
}
```

### 🖼️ Endpoints de Busca Visual (ChromaDB)

#### Adicionar Imagem à Busca Visual

```http
POST /api/v1/photos/visual-search/add
Content-Type: multipart/form-data

file: <arquivo de imagem>
description: "Descrição opcional"
tags: "tag1,tag2,tag3"
```

#### Busca Visual por Texto

```http
GET /api/v1/photos/visual-search/text?q=gato%20preto&limit=8
```

**Resposta:**

```json
{
  "query": "gato preto",
  "results": [
    {
      "image_path": "/path/to/image.jpg",
      "similarity": 0.87,
      "caption": "Um gato preto brilhante...",
      "tags": "animal,pet",
      "file_name": "cat.jpg"
    }
  ],
  "total_found": 5
}
```

#### Busca Reversa por Imagem

```http
POST /api/v1/photos/visual-search/image
Content-Type: multipart/form-data

file: <imagem de consulta>
limit: 8
```

#### Estatísticas da Busca Visual

```http
GET /api/v1/photos/visual-search/stats
```

**Resposta:**

```json
{
  "total_images": 42,
  "collection_name": "images",
  "embedding_dimensions": 512,
  "status": "active"
}
```

## 🔄 Migração para LangChain

### Contexto

O sistema foi migrado de uma implementação direta com Gemini API para uma arquitetura baseada em LangChain, oferecendo:

- **Múltiplos provedores**: OpenAI, Anthropic, Gemini, Local
- **Fallback automático**: Sistema robusto de contingência
- **OCR integrado**: Extração de texto de imagens
- **Melhor qualidade**: Prompts otimizados e processamento avançado

### Benefícios da Migração

- ✅ **Resiliência**: Não depende de um único provedor
- ✅ **Custo**: Opção de usar modelos locais gratuitos
- ✅ **Qualidade**: Melhor controle sobre geração de texto
- ✅ **Escalabilidade**: Fácil adição de novos provedores

## 📊 Monitoramento

### Script de Acompanhamento

```bash
# Monitor em tempo real (atualiza a cada 30s)
./monitor_progress.sh

# Ou via API
curl http://localhost:8000/api/v1/photos/processing/stats
```

### Verificar Status dos Serviços

```bash
# Status dos containers
docker compose ps

# Logs do worker
docker compose logs -f worker

# Logs do scheduler
docker compose logs -f scheduler
```

### Métricas de Performance

- **Processamento**: ~15 segundos por imagem
- **Busca**: < 100ms para consultas
- **Armazenamento**: Embeddings de 512 dimensões
- **OCR**: Suporte para 100+ idiomas

## 🔧 Configuração Avançada

### Variáveis de Ambiente

```bash
# Provedor de IA prioritário
AI_MODEL_TYPE=openai  # openai, anthropic, gemini, local

# Chaves de API
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Configurações do banco
DATABASE_URL=postgresql://user:pass@db:5432/photo_finder

# Scheduler
SCHEDULER_INTERVAL_SECONDS=10
```

### Modelos de IA Disponíveis

#### OpenAI (Recomendado)

- **Modelo**: GPT-4o-mini
- **Custo**: Baixo para descrições
- **Qualidade**: Excelente
- **Velocidade**: Rápida

#### Local (Gratuito)

- **Modelo**: GPT-2 ou DialoGPT
- **Custo**: Zero
- **Limitações**: Menos preciso, sem visão
- **Uso**: Desenvolvimento/testing

#### Anthropic

- **Modelo**: Claude 3 Haiku
- **Custo**: Médio
- **Qualidade**: Muito boa
- **Ética**: Foco em segurança

#### Google Gemini

- **Modelo**: Gemini 1.5 Flash
- **Custo**: Competitivo
- **Multimodal**: Bom para imagens
- **Integração**: Nativa do Google

## 🚨 Troubleshooting

### Problemas Comuns

#### 1. Worker não processa imagens

```bash
# Verificar logs
docker compose logs worker

# Verificar Redis
docker compose exec redis redis-cli ping
```

#### 2. Erro de API key

```text
Erro: OPENAI_API_KEY não configurada
Solução: Adicionar chave no .env ou usar modelo local
```

#### 3. Memória insuficiente

```text
Erro: CUDA out of memory
Solução: Usar modelo local menor ou aumentar RAM
```

#### 4. OCR não funciona

```bash
# Verificar instalação do Tesseract
docker compose exec app tesseract --version
```

### Logs e Debug

```bash
# Todos os logs
docker compose logs

# Logs específicos
docker compose logs app
docker compose logs worker
docker compose logs scheduler

# Limpar e reconstruir
docker compose down -v
docker compose up --build
```

## 📈 Performance e Escalabilidade

### Otimizações Implementadas

- **Processamento assíncrono**: RQ para background jobs
- **Embeddings eficientes**: pgvector para busca rápida
- **Cache inteligente**: Modelos de IA em disco
- **Fallback automático**: Sem pontos únicos de falha

### Limites e Recomendações

- **Imagens por upload**: Até 10 simultâneas
- **Tamanho máximo**: 10MB por imagem
- **Busca**: Até 50 resultados por consulta
- **Processamento**: ~100 imagens/hora (depende da API)

### Escalabilidade

- **Horizontal**: Múltiplos workers via Redis
- **Vertical**: Mais RAM para modelos maiores
- **Cloud**: Fácil migração para Kubernetes

## 🤝 Contribuição

### Desenvolvimento Local

```bash
# Instalar dependências
pip install -r requirements.txt

# Rodar testes
python test_ocr.py

# Verificar linting
# (adicionar ferramentas de lint se necessário)
```

### Estrutura de Código

- **API**: Endpoints RESTful em `/api/v1/`
- **Services**: Lógica de negócio isolada
- **Models**: SQLModel para type safety
- **Jobs**: RQ para processamento assíncrono

## 📄 Licença

Este projeto é distribuído sob a licença MIT. Veja o arquivo LICENSE para detalhes.

## 🙋 Suporte

Para questões, bugs ou sugestões:

1. Verifique os logs: `docker compose logs`
2. Teste com dados simples
3. Consulte a documentação da API: `/docs`

---

Desenvolvido usando FastAPI, LangChain e pgvector
