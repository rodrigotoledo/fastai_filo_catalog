from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import photos_router
from app.db.database import engine
from app.models.photo import Photo
from sqlmodel import SQLModel

# Criar tabelas no banco
SQLModel.metadata.create_all(bind=engine)

app = FastAPI(
    title="Photo Finder API",
    description="API for uploading and managing photos with AI capabilities",
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

app.include_router(photos_router, prefix="/photos", tags=["photos"])

@app.get("/")
def root():
    return {"message": "Photo Finder API - backend base"}
