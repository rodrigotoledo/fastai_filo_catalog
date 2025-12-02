from fastapi import APIRouter, Depends, HTTPException, Query, Form
from sqlalchemy.orm import Session
from typing import List, Optional
from app.db.database import get_db
from app.services.client_service import ClientService
from app.models.client import Client
from app.schemas.client import (
    ClientCreate, ClientUpdate, ClientResponse,
    PaginatedClientsResponse
)

router = APIRouter()

@router.post("/", response_model=ClientResponse)
def create_client(
    client: ClientCreate,
    db: Session = Depends(get_db)
):
    """
    Criar novo cliente com endereços
    """
    client_service = ClientService(db)
    try:
        return client_service.create_client(client)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/", response_model=PaginatedClientsResponse)
def get_clients(
    page: int = Query(1, ge=1, description="Página atual"),
    page_size: int = Query(12, ge=1, le=100, description="Itens por página"),
    search: Optional[str] = Query(None, description="Buscar por nome, email ou CPF"),
    db: Session = Depends(get_db)
):
    """
    Listar clientes com paginação e busca
    """
    client_service = ClientService(db)
    return client_service.get_clients(page=page, page_size=page_size, search=search)

@router.get("/{client_id}", response_model=ClientResponse)
def get_client(
    client_id: int,
    db: Session = Depends(get_db)
):
    """
    Buscar cliente específico por ID
    """
    client_service = ClientService(db)
    client = client_service.get_client(client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Cliente não encontrado")
    return client

@router.put("/{client_id}", response_model=ClientResponse)
def update_client(
    client_id: int,
    client: ClientUpdate,
    db: Session = Depends(get_db)
):
    """
    Atualizar cliente e seus endereços
    """
    client_service = ClientService(db)
    try:
        return client_service.update_client(client_id, client)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/{client_id}")
def delete_client(
    client_id: int,
    db: Session = Depends(get_db)
):
    """
    Remover cliente
    """
    client_service = ClientService(db)
    success = client_service.delete_client(client_id)
    if not success:
        raise HTTPException(status_code=404, detail="Cliente não encontrado")
    return {"message": "Cliente removido com sucesso"}

@router.patch("/{client_id}/toggle-status", response_model=ClientResponse)
def toggle_client_status(
    client_id: int,
    db: Session = Depends(get_db)
):
    """
    Ativar/desativar cliente
    """
    client_service = ClientService(db)
    try:
        return client_service.toggle_client_status(client_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/populate", response_model=List[ClientResponse])
def populate_clients(
    count: int = Form(10, description="Número de clientes a criar (1-50)", ge=1, le=50),
    db: Session = Depends(get_db)
):
    """
    Criar múltiplos clientes fake com endereços aleatórios (1-3 por cliente)
    """
    client_service = ClientService(db)
    try:
        return client_service.populate_clients(count)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
