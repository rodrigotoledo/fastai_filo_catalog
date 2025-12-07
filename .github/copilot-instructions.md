# Photo Finder Backend - AI Coding Guidelines

## Architecture Overview
- **Framework**: FastAPI with SQLModel (SQLAlchemy + Pydantic) for ORM and validation
- **Database**: PostgreSQL with pgvector extension for AI embeddings
- **Queue System**: Redis + RQ for asynchronous photo processing
- **AI**: CLIP (ViT-B/32) for image embeddings and descriptions
- **Containerization**: Docker Compose with separate services for app, db, redis

## Project Structure
- `app/models/`: SQLModel table definitions (Client, ClientAddress, Photo)
- `app/schemas/`: Pydantic schemas for API requests/responses
- `app/services/`: Business logic layer (ClientService, PhotoService, AIService)
- `app/api/`: FastAPI routers with dependency injection
- `app/db/`: Database connection and session management
- `alembic/`: Database migrations
- `app/jobs/`: RQ job definitions for background processing

## Key Patterns
- **Models**: Use SQLModel with relationships; foreign keys use singular table names (e.g., `client_id: int = Field(foreign_key="client.id")`)
- **Services**: Instantiate with DB session; handle validation and business rules
- **API**: Use dependency injection for DB sessions; return schema models, not table models
- **Migrations**: Use `alembic revision --autogenerate -m "message"` after model changes
- **Jobs**: Enqueue via RQ; process in background with `worker.py`

## Development Workflow
- **Start services**: `docker compose up` (includes db, redis, app)
- **Run migrations**: `alembic upgrade head`
- **Process photos**: Run `python scheduler.py` to enqueue unprocessed photos
- **Start worker**: `python worker.py` for background AI processing
- **API docs**: Visit `http://localhost:8000/docs` for Swagger UI

## Database Conventions
- **Table names**: Plural (clients, client_addresses, photos)
- **Embeddings**: 512-dimensional vectors stored in pgvector
- **Relationships**: Use SQLModel Relationship with back_populates for bidirectional access

## AI Integration
- **CLIP Model**: Cached locally in `cache/` directory
- **Processing**: Images processed asynchronously via RQ queues
- **Embeddings**: Used for similarity search (future feature)
- **Fallback**: Simple hash-based embeddings if CLIP fails

## Common Tasks
- **Add new client field**: Update `Client` model → `alembic revision --autogenerate` → update schemas
- **New API endpoint**: Add to appropriate router in `app/api/`, create service method
- **Background job**: Define in `app/jobs/`, enqueue from service or scheduler
- **Environment vars**: Use `.env` file for DATABASE_URL, etc.
- **Populate test data**: POST `/api/v1/clients/populate?count=N` or `/api/v1/photos/populate?term=TERM&count=N`

## File Examples
- **Model relationship**: See `Client.addresses` in `app/models/client.py`
- **Service pattern**: `ClientService.create_client()` validates and saves with addresses
- **API dependency**: `db: Session = Depends(get_db)` in router functions
- **Migration example**: Rename tables to plural in future migration
