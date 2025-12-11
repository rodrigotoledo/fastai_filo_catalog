from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import ForeignKey, Column
from typing import Optional, List
from datetime import datetime
from pgvector.sqlalchemy import Vector

class ClientAddress(SQLModel, table=True):
    __tablename__ = 'client_addresses'
    id: Optional[int] = Field(default=None, primary_key=True)
    client_id: int = Field(foreign_key="clients.id")
    type: str = Field(default="Pessoal")  # Pessoal, Comercial, etc.
    street: str
    number: str
    complement: Optional[str] = None
    neighborhood: str
    city: str
    state: str
    zip_code: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationship back to client
    client: Optional["Client"] = Relationship(back_populates="addresses")

class Client(SQLModel, table=True):
    __tablename__ = 'clients'
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    nickname: Optional[str] = None
    email: str = Field(unique=True, index=True)
    phone: Optional[str] = None
    cpf: Optional[str] = Field(unique=True, index=True)
    rg: Optional[str] = None
    birth_date: Optional[datetime] = None
    is_active: bool = Field(default=True)
    processed: bool = Field(default=False)

    # Embedding vetorial para busca semântica (1536 dimensões - OpenAI)
    embedding: Optional[list[float]] = Field(default=None, sa_column=Column(Vector(1536)))

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationship to addresses
    addresses: List[ClientAddress] = Relationship(back_populates="client", cascade_delete=True)
