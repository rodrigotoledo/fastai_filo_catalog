from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from app.models.photo import Photo

class PhotoResponse(BaseModel):
    id: int
    filename: str
    original_filename: str
    file_path: str
    file_size: int
    content_type: str
    uploaded_at: datetime
    processed: bool

    class Config:
        from_attributes = True

class PaginatedPhotosResponse(BaseModel):
    photos: List[PhotoResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool
