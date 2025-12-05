from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime
from sqlalchemy import Column
from pgvector.sqlalchemy import Vector

class Photo(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    original_filename: str
    file_path: str
    file_size: int
    content_type: str
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    processed: bool = Field(default=False)
    # Campos para IA
    embedding: Optional[list[float]] = Field(default=None, sa_column=Column(Vector(512)))
    description: Optional[str] = None
    user_description: Optional[str] = None  # Prompt descritivo fornecido pelo usuário
    gemini_file_id: Optional[str] = None  # ID do arquivo no Gemini File Search Store
