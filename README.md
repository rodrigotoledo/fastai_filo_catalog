# Photo Finder API

API RESTful para upload, armazenamento e gerenciamento de fotos construída com FastAPI. Inclui funcionalidades de paginação, validação de arquivos e suporte a PostgreSQL com pgvector para futuras implementações de IA.

## 🚀 Funcionalidades

- **Upload múltiplo de fotos** com validação de tipo de arquivo
- **Paginação inteligente** (página/tamanho personalizado)
- **Servir arquivos estáticos** diretamente via API
- **Banco PostgreSQL** com suporte a vetores (pgvector)
- **Redis** para cache e filas assíncronas
- **Documentação automática** via Swagger/OpenAPI
- **CORS configurado** para frontend (Next.js)
- **Docker completo** para desenvolvimento

## 📋 Requisitos

- Python 3.8+
- Docker & Docker Compose
- PostgreSQL 13+ (via Docker)
- Redis (via Docker)


## 🛠️ Instalação e Setup


### 1. Clone o repositório


```bash
git clone <seu-repositorio>
cd photo-finder/backend

```

### 2. Configure o ambiente


```bash
# Crie ambiente virtual (opcional)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou .venv\Scripts\activate no Windows

# Instale dependências
pip install -r requirements.txt

```

### 3. Configure variáveis de ambiente

```bash
cp .env.example .env
# Edite .env conforme necessário
```

### 4. Inicie os serviços com Docker

```bash
docker compose up -d
```

Isso iniciará:

- **PostgreSQL** na porta 5432
- **Redis** na porta 6379
- **Aplicação FastAPI** na porta 8000

## 🗄️ Banco de Dados

### Migrações

```bash
# Criar nova migração
./alembic.sh revision --autogenerate -m "Descrição"

# Aplicar migrações
./alembic.sh upgrade head
```

### Ou via Docker

```bash
docker compose exec app alembic upgrade head
```

## 🚀 Execução

### Ambiente de Desenvolvimento

```bash
# Via Python
python run.py

# Ou diretamente
uvicorn app.main:app --reload
```

### Acesse

- **API**: [http://localhost:8000](http://localhost:8000)
- **Documentação**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## 📚 API Endpoints

### Upload de Fotos

```http
POST /photos/upload
```

- **Body**: `multipart/form-data` com campo `files[]`
- **Suporte**: Múltiplas imagens (JPEG, PNG, etc.)
- **Resposta**: Lista de fotos criadas

### Listar Fotos (com paginação)

```http
GET /photos/?page=1&page_size=10
```

**Parâmetros:**

- `page` (int, ≥1): Número da página
- `page_size` (int, 1-100): Itens por página (padrão: 10)

**Resposta:**

```json
{
  "photos": [...],
  "total": 150,
  "page": 1,
  "page_size": 10,
  "total_pages": 15,
  "has_next": true,
  "has_prev": false
}
```

### Obter Foto Específica

```http
GET /photos/{photo_id}
```

### Servir Arquivo de Foto

```http
GET /photos/file/{photo_id}
```

Retorna o arquivo binário da imagem.

## 🧪 Testes

### Upload de teste

```bash
# Baixar imagem de teste
curl -L -s "https://loremflickr.com/400/300/cat" --output test.jpg

# varios arquivos
for i in {2..20}; do curl -L -s "https://loremflickr.com/800/600/cat?random=$i" --output cat_image$i.jpg; done

# Fazer upload
curl -X POST -F "files=@test.jpg" http://localhost:8000/photos/upload
```

### Listar fotos

```bash
curl "http://localhost:8000/photos/?page=1&page_size=5"
```

## 📁 Estrutura do Projeto

```
backend/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── photos.py          # Endpoints de fotos
│   ├── models/
│   │   ├── __init__.py
│   │   └── photo.py           # Modelo Photo (SQLModel)
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── photo.py           # Schemas Pydantic
│   ├── services/
│   │   ├── __init__.py
│   │   └── photo_service.py   # Lógica de negócio
│   ├── db/
│   │   └── database.py        # Configuração DB
│   └── main.py                # App FastAPI
├── alembic/                   # Migrações DB
├── uploads/                   # Arquivos enviados
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── .env.example
└── README.md
```

## 🐳 Docker

### Desenvolvimento
```bash
# Subir todos os serviços
docker compose up -d

# Ver logs
docker compose logs -f app

# Executar comandos no container
docker compose exec app bash
```

### Ambiente de Produção
O `Dockerfile` está configurado para produção com usuário não-root e permissões adequadas.

## 🔧 Principais Dependências

- **FastAPI**: Framework web assíncrono
- **SQLModel**: ORM com Pydantic
- **PostgreSQL + pgvector**: DB com suporte a vetores
- **Redis**: Cache e filas
- **Alembic**: Migrações de banco
- **python-multipart**: Upload de arquivos
- **aiofiles**: Manipulação assíncrona de arquivos

## 🚀 Deploy

### Ambiente Local
```bash
docker compose up -d
python run.py
```

### Produção
```bash
# Build da imagem
docker build -t photo-finder .

# Run com compose
docker compose -f docker-compose.prod.yml up -d
```

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## 📄 Licença

MIT License - veja o arquivo LICENSE para detalhes.
