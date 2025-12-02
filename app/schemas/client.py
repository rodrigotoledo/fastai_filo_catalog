from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime

# Documents schema
class ClientDocuments(BaseModel):
    cpf: Optional[str] = None
    rg: Optional[str] = None
    birth_date: Optional[str] = None  # String format for API

# Address schemas
class ClientAddressBase(BaseModel):
    type: str = "Pessoal"
    street: str
    number: str
    complement: Optional[str] = None
    neighborhood: str
    city: str
    state: str
    zip_code: str

class ClientAddressCreate(ClientAddressBase):
    pass

class ClientAddressUpdate(ClientAddressBase):
    pass

class ClientAddressResponse(ClientAddressBase):
    id: int
    client_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# Client schemas
class ClientBase(BaseModel):
    name: str
    nickname: Optional[str] = None
    email: EmailStr
    phone: Optional[str] = None
    documents: ClientDocuments = Field(default_factory=ClientDocuments)

class ClientCreate(ClientBase):
    addresses: List[ClientAddressCreate] = []

class ClientUpdate(ClientBase):
    addresses: List[ClientAddressCreate] = []  # Pode incluir IDs existentes ou novos

class ClientResponse(ClientBase):
    id: int
    created_at: datetime
    updated_at: datetime
    addresses: List[ClientAddressResponse] = []

    class Config:
        from_attributes = True

# Paginated response
class PaginatedClientsResponse(BaseModel):
    clients: List[ClientResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool
