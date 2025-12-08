from fastapi import APIRouter, Depends, HTTPException, Query, Form, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Optional, Annotated
from pydantic import BaseModel
from app.db.database import get_db
from app.services.client_service import ClientService
from app.services.document_parser_service import DocumentParserService
from app.services.ai_service import AIService
from app.models.client import Client
from app.schemas.client import (
    ClientCreate, ClientUpdate, ClientResponse,
    PaginatedClientsResponse
)

# Modelo Pydantic simples para arquivo validado
class ValidatedFile(BaseModel):
    file: UploadFile

# Dependência para arquivo validado com validação explícita
async def get_validated_file(file: UploadFile) -> ValidatedFile:
    """Dependência FastAPI que valida arquivo explicitamente."""
    # Validar extensão do arquivo
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nome do arquivo é obrigatório")

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

    # Validar tamanho do arquivo (10MB)
    content = await file.read()
    file_size = len(content)

    if file_size > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(status_code=400, detail="Arquivo muito grande. Máximo: 10MB")

    # Resetar ponteiro do arquivo para que possa ser lido novamente
    import io
    file.file = io.BytesIO(content)

    # Retornar arquivo validado
    return ValidatedFile(file=file)

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

@router.get("/search-similar", response_model=List[dict])
def search_similar_clients(
    q: str = Query(..., description="Texto para busca semântica de clientes (ex: 'cliente de São Paulo com email gmail')", min_length=3),
    limit: int = Query(10, description="Número máximo de resultados", ge=1, le=50),
    db: Session = Depends(get_db)
):
    """
    Busca clientes usando similaridade semântica vetorial.

    Esta endpoint permite encontrar clientes através de busca semântica,
    não apenas busca textual tradicional. Por exemplo:
    - "cliente de São Paulo" - encontra clientes com endereço em SP
    - "pessoa com email gmail" - encontra clientes com emails @gmail.com
    - "cliente nascido em 1980" - encontra clientes nascidos nessa década

    **Como funciona:**
    1. Gera embedding vetorial da query usando IA (CLIP)
    2. Compara com embeddings armazenados dos clientes
    3. Retorna clientes ordenados por similaridade

    **Limitações atuais:**
    - Apenas clientes criados via upload de documento têm embeddings
    - Clientes criados manualmente não têm embeddings vetoriais
    """
    client_service = ClientService(db)
    results = client_service.search_similar_clients(q, limit)

    return results

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
    validated_file: Annotated[ValidatedFile, Depends(get_validated_file)],
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
    # Arquivo já validado pela dependência - obter dados
    file = validated_file.file
    file_extension = file.filename.split('.')[-1].lower() if '.' in file.filename else ''

    # Ler conteúdo do arquivo (já validado quanto ao tamanho)
    content = await file.read()
    file_size = len(content)

    try:
        # Inicializar serviços
        ai_service = AIService()
        document_parser = DocumentParserService(ai_service)

        # Processar documento diretamente dos bytes (sem arquivo temporário)
        extracted_data = document_parser.parse_document_from_bytes(content, file.filename, extraction_prompt)

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
            client_service = ClientService(db)
            created_client = client_service.create_client_from_extracted_data(extracted_data)

            response["client_created"] = True
            response["created_client"] = created_client
        else:
            response["client_created"] = False

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")
