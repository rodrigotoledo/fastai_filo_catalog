from fastapi import APIRouter, Depends, HTTPException, Query, Form, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Optional
from app.db.database import get_db
from app.services.client_service import ClientService
from app.services.document_parser_service import DocumentParserService
from app.services.ai_service import AIService
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

@router.post("/upload-document", response_model=dict)
async def upload_document(
    file: UploadFile = File(..., description="Arquivo a ser processado (PDF, DOCX, imagem, CSV, XLSX, MD, TXT)"),
    create_client: bool = Form(False, description="Se verdadeiro, cria o cliente automaticamente se os dados forem válidos"),
    extraction_prompt: Optional[str] = Form(None, description="Prompt personalizado para orientar a extração de dados (ex: 'Extraia nome, email e telefone do currículo')"),
    db: Session = Depends(get_db)
):
    """
    Faz upload e processa um documento para extrair dados de cliente usando IA.

    **Formatos suportados:**
    - PDF (.pdf)
    - Word (.docx)
    - Imagens (.png, .jpg, .jpeg, .tiff, .bmp) - usa OCR
    - CSV (.csv)
    - Excel (.xlsx, .xls)
    - Markdown (.md)
    - Texto (.txt)

    **Processamento:**
    1. Extrai texto do documento usando bibliotecas apropriadas
    2. Usa IA para identificar e extrair dados estruturados do cliente
    3. Valida os dados extraídos
    4. Opcionalmente cria o cliente se solicitado e dados forem válidos

    **Prompt Personalizado:**
    Você pode fornecer um `extraction_prompt` para orientar a extração:
    - "Extraia nome, email e telefone"
    - "Este documento contém informações de cliente: nome, CPF, endereço"
    - "Procure por dados pessoais em português brasileiro"

    Se não fornecido, usa extração automática padrão.

    **Resposta:**
    Retorna os dados extraídos e informações sobre o processamento.
    Se create_client=True e dados válidos, também retorna o cliente criado.
    """
    # Validar tipo de arquivo
    allowed_extensions = {
        'pdf', 'docx', 'png', 'jpg', 'jpeg', 'tiff', 'bmp',
        'csv', 'xlsx', 'xls', 'md', 'txt'
    }

    file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de arquivo não suportado. Use: {', '.join(allowed_extensions)}"
        )

    # Validar tamanho do arquivo (máximo 10MB)
    file_size = 0
    content = await file.read()
    file_size = len(content)

    if file_size > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(status_code=400, detail="Arquivo muito grande. Máximo: 10MB")

    # Salvar arquivo temporariamente
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
        temp_file.write(content)
        temp_file_path = temp_file.name

    try:
        # Inicializar serviços
        ai_service = AIService()
        document_parser = DocumentParserService(ai_service)

        # Processar documento
        extracted_data = document_parser.parse_document(temp_file_path, file.filename, extraction_prompt)

        # Validar dados extraídos
        validation_errors = document_parser.validate_extracted_data(extracted_data)

        response = {
            "filename": file.filename,
            "file_size": file_size,
            "file_type": file_extension.upper(),
            "extracted_data": extracted_data,
            "validation_errors": validation_errors,
            "is_valid": len(validation_errors) == 0,
            "processing_status": "success"
        }

        # Criar cliente se solicitado e dados forem válidos
        if create_client and len(validation_errors) == 0 and extracted_data.get('name'):
            try:
                client_service = ClientService(db)

                # Sempre gerar email se não houver um válido
                email = extracted_data.get('email')
                if not email:
                    # Usar nome para gerar email único
                    name_clean = extracted_data['name'].lower().replace(' ', '.').replace('ç', 'c').replace('ã', 'a').replace('õ', 'o')
                    name_clean = ''.join(c for c in name_clean if c.isalnum() or c == '.')
                    email = f"{name_clean}@temp.document"

                client_data = ClientCreate(
                    name=extracted_data['name'],
                    email=email,
                    phone=extracted_data.get('phone'),
                    documents={
                        'cpf': extracted_data.get('cpf'),
                        'birth_date': extracted_data.get('date_of_birth')
                    },
                    addresses=[]
                )

                # Adicionar endereço se disponível (apenas campos obrigatórios)
                address_data = extracted_data.get('address', {})
                if address_data.get('street') or address_data.get('city'):
                    from app.schemas.client import ClientAddressCreate
                    address = ClientAddressCreate(
                        street=address_data.get('street') or "Endereço não informado",
                        number=address_data.get('number') or "S/N",
                        neighborhood=address_data.get('neighborhood') or "Centro",
                        city=address_data.get('city') or "São Paulo",
                        state=address_data.get('state') or "SP",
                        zip_code=address_data.get('postal_code') or "00000-000"
                    )
                    client_data.addresses = [address]

                # Criar cliente
                created_client = client_service.create_client(client_data)

                response["client_created"] = True
                response["created_client"] = created_client

            except Exception as e:
                response["client_creation_error"] = str(e)
                response["client_created"] = False
        else:
            response["client_created"] = False

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")

    finally:
        # Limpar arquivo temporário
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
