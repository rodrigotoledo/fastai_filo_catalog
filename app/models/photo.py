from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

class Photo(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    original_filename: str
    file_path: str
    file_size: int
    content_type: str
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    processed: bool = Field(default=False)
    # Campos para IA (a serem adicionados depois)
    # embedding: Optional[list[float]] = Field(default=None, sa_column=Column(Vector(768)))
    # description: Optional[str] = None
