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
    search: Optional[str] = Query(None, description="Buscar por nome, email ou CPF (busca tradicional LIKE)"),
    db: Session = Depends(get_db)
):
    """
    Listar clientes com paginação e busca tradicional

    Este endpoint faz busca tradicional (LIKE) por substring em nome, email ou CPF.
    Use este endpoint para navegação normal da lista de clientes.

    Para busca inteligente por similaridade semântica, use /search/similar
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

@router.post("/{client_id}/process", response_model=ClientResponse)
def process_client(
    client_id: int,
    user_description: Optional[str] = Form(None, description="Descrição adicional para processamento com IA"),
    db: Session = Depends(get_db)
):
    """
    Processa um cliente com IA para gerar embedding e permitir busca por similaridade
    """
    client_service = ClientService(db)
    try:
        return client_service.process_client(client_id, user_description)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/search/similar", response_model=PaginatedClientsResponse)
def search_similar_clients(
    q: Optional[str] = Query(None, description="Texto para busca por similaridade"),
    client_id: Optional[int] = Query(None, description="ID do cliente de referência"),
    page: int = Query(1, ge=1, description="Página atual"),
    page_size: int = Query(12, ge=1, le=100, description="Itens por página"),
    db: Session = Depends(get_db)
):
    """
    Busca clientes similares por texto ou por outro cliente usando IA com paginação

    Retorna o MESMO FORMATO do endpoint básico GET /, mas com similarity_score adicionado
    aos objetos ClientResponse quando aplicável.

    **Lógica de Filtragem Inteligente:**
    - **Termos específicos** (ex: "Maria"): retorna apenas correspondências exatas (score = 1.0)
    - **Termos genéricos** (ex: "cliente"): retorna top 6 resultados mais similares
    - **Busca por cliente**: encontra clientes similares a um cliente de referência

    **Formato de Resposta:** Mesmo que GET / (PaginatedClientsResponse)
    - clients: Array de ClientResponse (com similarity_score opcional)
    - total, page, page_size, total_pages, has_next, has_prev

    Use este endpoint quando o usuário fizer uma busca específica no frontend.
    """
    if not q and not client_id:
        raise HTTPException(status_code=400, detail="Deve fornecer 'q' (texto) ou 'client_id'")

    client_service = ClientService(db)
    try:
        # Obter resultados da busca por similaridade
        similar_results = client_service.search_similar_clients(q, client_id, page, page_size)

        # Converter para o formato padrão, adicionando similarity_score
        clients_with_scores = []
        for item in similar_results["results"]:
            client_response = item["client"]
            client_response.similarity_score = item["similarity_score"]
            clients_with_scores.append(client_response)

        return {
            "clients": clients_with_scores,
            "total": similar_results["total"],
            "page": similar_results["page"],
            "page_size": similar_results["page_size"],
            "total_pages": similar_results["total_pages"],
            "has_next": similar_results["has_next"],
            "has_prev": similar_results["has_prev"]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
