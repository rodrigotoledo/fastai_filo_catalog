from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import photos_router, clients_router
from app.db.database import engine
from app.models.photo import Photo
from app.models.client import Client, ClientAddress
from sqlmodel import SQLModel

# Criar tabelas no banco (remover em produção, usar migrations)
# SQLModel.metadata.create_all(bind=engine)

app = FastAPI(
    title="Photo Finder API",
    description="API for uploading and managing photos with AI capabilities and client management",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend Next.js
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(photos_router, prefix="/api/v1/photos", tags=["photos"])
app.include_router(clients_router, prefix="/api/v1/clients", tags=["clients"])

@app.get("/")
def root():
    return {"message": "Photo Finder API - backend with photos and clients"}
