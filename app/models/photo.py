from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime
from sqlalchemy import Column
from pgvector.sqlalchemy import Vector

class Photo(SQLModel, table=True):
    __tablename__ = 'photos'
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    original_filename: str
    file_path: str
    file_size: int
    content_type: str
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    processed: bool = Field(default=False)
    # Campos para IA (1536 dimensões - OpenAI)
    embedding: Optional[list[float]] = Field(default=None, sa_column=Column(Vector(1536)))
    embedding: Optional[list[float]] = Field(default=None, sa_column=Column(Vector(1536)))
    description: Optional[str] = None
    user_description: Optional[str] = None  # Prompt descritivo fornecido pelo usuário
    gemini_file_id: Optional[str] = None  # ID do arquivo no Gemini File Search Store
    image_data: Optional[bytes] = None  # Raw image bytes for CLIP processing
