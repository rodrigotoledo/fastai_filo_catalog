from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_, text
from typing import List, Optional, Dict
from app.models.client import Client, ClientAddress
from app.schemas.client import (
    ClientCreate, ClientUpdate, ClientResponse,
    ClientAddressCreate, ClientAddressResponse, ClientDocuments
)
import re
from datetime import datetime
from app.services.document_parser_service import DocumentParserService
from app.services.ai_service import AIService
import logging

logger = logging.getLogger(__name__)

class ClientService:
    def __init__(self, db: Session):
        self.db = db
        self.ai_service = AIService()

    def _validate_cpf(self, cpf: str) -> bool:
        """Valida CPF brasileiro"""
        if not cpf:
            return True  # CPF opcional

        # Remove caracteres não numéricos
        cpf = re.sub(r'\D', '', cpf)

        if len(cpf) != 11:
            return False

        # Verifica se todos os dígitos são iguais
        if cpf == cpf[0] * 11:
            return False

        # Calcula primeiro dígito verificador
        soma = sum(int(cpf[i]) * (10 - i) for i in range(9))
        resto = (soma * 10) % 11
        if resto == 10:
            resto = 0
        if resto != int(cpf[9]):
            return False

        # Calcula segundo dígito verificador
        soma = sum(int(cpf[i]) * (11 - i) for i in range(10))
        resto = (soma * 10) % 11
        if resto == 10:
            resto = 0
        if resto != int(cpf[10]):
            return False

        return True

    def _format_cpf(self, cpf: str) -> str:
        """Formata CPF para exibição"""
        if not cpf:
            return cpf
        cpf = re.sub(r'\D', '', cpf)
        return f"{cpf[:3]}.{cpf[3:6]}.{cpf[6:9]}-{cpf[9:]}"

    def _format_phone(self, phone: str) -> str:
        """Formata telefone para exibição"""
        if not phone:
            return phone
        phone = re.sub(r'\D', '', phone)
        if len(phone) == 11:  # Celular com DDD
            return f"({phone[:2]}) {phone[2:7]}-{phone[7:]}"
        elif len(phone) == 10:  # Fixo com DDD
            return f"({phone[:2]}) {phone[2:6]}-{phone[6:]}"
        return phone

    def _parse_birth_date(self, birth_date_str: str) -> Optional[datetime]:
        """Converte string de data para datetime"""
        if not birth_date_str:
            return None
        try:
            # Tentar formato ISO primeiro
            return datetime.fromisoformat(birth_date_str)
        except ValueError:
            # Tentar formato brasileiro DD/MM/YYYY
            try:
                return datetime.strptime(birth_date_str, "%d/%m/%Y")
            except ValueError:
                return None

    def _client_to_response(self, client: Client) -> ClientResponse:
        """Converte Client model para ClientResponse schema"""
        return ClientResponse(
            id=client.id,
            name=client.name,
            nickname=client.nickname,
            email=client.email,
            phone=self._format_phone(client.phone) if client.phone else None,
            documents=ClientDocuments(
                cpf=self._format_cpf(client.cpf) if client.cpf else None,
                rg=client.rg,
                birth_date=client.birth_date.strftime("%Y-%m-%d") if client.birth_date else None
            ),
            created_at=client.created_at,
            updated_at=client.updated_at,
            addresses=[
                ClientAddressResponse(
                    id=addr.id,
                    client_id=addr.client_id,
                    type=addr.type,
                    street=addr.street,
                    number=addr.number,
                    complement=addr.complement,
                    neighborhood=addr.neighborhood,
                    city=addr.city,
                    state=addr.state,
                    zip_code=addr.zip_code,
                    created_at=addr.created_at,
                    updated_at=addr.updated_at
                ) for addr in client.addresses
            ]
        )

    def _format_cpf(self, cpf: str) -> str:
        """Formata CPF para exibição"""
        if not cpf:
            return cpf
        cpf = re.sub(r'\D', '', cpf)
        return f"{cpf[:3]}.{cpf[3:6]}.{cpf[6:9]}-{cpf[9:]}"

    def create_client(self, client_data: ClientCreate) -> Client:
        """Cria um novo cliente com endereços"""
        # Validar CPF se fornecido
        if client_data.documents.cpf and not self._validate_cpf(client_data.documents.cpf):
            raise ValueError("CPF inválido")

        # Verificar se email já existe
        existing_client = self.db.query(Client).filter(Client.email == client_data.email).first()
        if existing_client:
            raise ValueError("Email já cadastrado")

        # Verificar se CPF já existe (se fornecido)
        if client_data.documents.cpf:
            existing_cpf = self.db.query(Client).filter(Client.cpf == client_data.documents.cpf).first()
            if existing_cpf:
                raise ValueError("CPF já cadastrado")

        # Criar cliente
        client_dict = client_data.model_dump(exclude={'addresses', 'documents'})
        client_dict.update({
            'cpf': client_data.documents.cpf,
            'rg': client_data.documents.rg,
            'birth_date': self._parse_birth_date(client_data.documents.birth_date)
        })
        client = Client(**client_dict)

        # Adicionar endereços
        for addr_data in client_data.addresses:
            address = ClientAddress(**addr_data.model_dump())
            client.addresses.append(address)

        self.db.add(client)
        self.db.commit()
        self.db.refresh(client)

        return self._client_to_response(client)

    def update_client(self, client_id: int, client_data: ClientUpdate) -> Client:
        """Atualiza cliente e seus endereços"""
        client = self.db.query(Client).filter(Client.id == client_id).first()
        if not client:
            raise ValueError("Cliente não encontrado")

        # Validar CPF se fornecido
        if client_data.documents.cpf and not self._validate_cpf(client_data.documents.cpf):
            raise ValueError("CPF inválido")

        # Verificar se email já existe (exceto para este cliente)
        existing_email = self.db.query(Client).filter(
            Client.email == client_data.email,
            Client.id != client_id
        ).first()
        if existing_email:
            raise ValueError("Email já cadastrado")

        # Verificar se CPF já existe (se fornecido)
        if client_data.documents.cpf:
            existing_cpf = self.db.query(Client).filter(
                Client.cpf == client_data.documents.cpf,
                Client.id != client_id
            ).first()
            if existing_cpf:
                raise ValueError("CPF já cadastrado")

        # Atualizar dados do cliente
        client_dict = client_data.model_dump(exclude={'addresses', 'documents'})
        client_dict.update({
            'cpf': client_data.documents.cpf,
            'rg': client_data.documents.rg,
            'birth_date': self._parse_birth_date(client_data.documents.birth_date)
        })

        for field, value in client_dict.items():
            setattr(client, field, value)

        # Atualizar endereços
        # Primeiro, marcar todos os endereços existentes para remoção
        existing_addresses = {addr.id: addr for addr in client.addresses}

        # Processar endereços da request
        new_addresses = []
        for addr_data in client_data.addresses:
            addr_dict = addr_data.model_dump()

            # Se tem ID, é atualização
            if 'id' in addr_dict and addr_dict['id'] and addr_dict['id'] in existing_addresses:
                addr = existing_addresses[addr_dict['id']]
                for field, value in addr_dict.items():
                    if field != 'id':
                        setattr(addr, field, value)
                del existing_addresses[addr_dict['id']]
            else:
                # Novo endereço
                new_addr = ClientAddress(**{k: v for k, v in addr_dict.items() if k != 'id'})
                new_addresses.append(new_addr)

        # Remover endereços não incluídos na atualização
        for addr in existing_addresses.values():
            self.db.delete(addr)

        # Adicionar novos endereços
        for addr in new_addresses:
            client.addresses.append(addr)

        self.db.commit()
        self.db.refresh(client)

        return self._client_to_response(client)

    def get_client(self, client_id: int) -> Optional[ClientResponse]:
        """Busca cliente por ID com endereços"""
        client = self.db.query(Client).options(
            joinedload(Client.addresses)
        ).filter(Client.id == client_id).first()

        return self._client_to_response(client) if client else None

    def get_clients(self, page: int = 1, page_size: int = 12, search: str = None) -> dict:
        """Lista clientes com paginação e busca"""
        if page < 1:
            page = 1

        query = self.db.query(Client).options(joinedload(Client.addresses))

        # Aplicar busca se fornecida
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                or_(
                    Client.name.ilike(search_term),
                    Client.nickname.ilike(search_term),
                    Client.email.ilike(search_term),
                    Client.cpf.ilike(search_term)
                )
            )

        total = query.count()
        skip = (page - 1) * page_size
        clients = query.order_by(Client.created_at.desc()).offset(skip).limit(page_size).all()

        total_pages = (total + page_size - 1) // page_size

        return {
            "clients": [self._client_to_response(client) for client in clients],
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }

    def populate_clients(self, count: int = 10) -> List[ClientResponse]:
        """
        Cria múltiplos clientes fake com endereços aleatórios (1-3 por cliente)
        """
        if count < 1 or count > 50:
            raise ValueError("Count must be between 1 and 50")

        import random
        from faker import Faker

        fake = Faker('pt_BR')  # Usar dados brasileiros

        clients = []

        try:
            for i in range(count):
                # Gerar dados do cliente
                name = fake.name()
                email = fake.email()
                phone = fake.phone_number()
                cpf = fake.cpf()
                birth_date = fake.date_of_birth(minimum_age=18, maximum_age=80)

                # Verificar se CPF já existe
                existing_cpf = self.db.query(Client).filter(Client.cpf == cpf).first()
                if existing_cpf:
                    cpf = fake.cpf()  # Gerar outro CPF

                # Verificar se email já existe
                existing_email = self.db.query(Client).filter(Client.email == email).first()
                if existing_email:
                    email = fake.email()  # Gerar outro email

                # Criar cliente
                client = Client(
                    name=name,
                    email=email,
                    phone=phone,
                    cpf=cpf,
                    birth_date=birth_date,
                    is_active=random.choice([True, True, True, False])  # 75% chance de ativo
                )

                self.db.add(client)
                self.db.flush()  # Para obter o ID

                # Criar 1-3 endereços aleatórios
                num_addresses = random.randint(1, 3)
                addresses = []

                for j in range(num_addresses):
                    address = ClientAddress(
                        client_id=client.id,
                        type=random.choice(["Pessoal", "Comercial", "Cobrança"]),
                        street=fake.street_name(),
                        number=str(random.randint(1, 9999)),
                        complement=f"Apt {random.randint(1, 999)}" if random.choice([True, False]) else None,
                        neighborhood=fake.bairro(),
                        city=fake.city(),
                        state=fake.estado_sigla(),
                        zip_code=fake.postcode()
                    )
                    addresses.append(address)
                    self.db.add(address)

                clients.append(client)

            # Commit de todos os dados de uma vez
            self.db.commit()

            # Refresh de todos os clientes para carregar addresses
            for client in clients:
                self.db.refresh(client, ['addresses'])

        except Exception as e:
            self.db.rollback()
            raise ValueError(f"Erro ao criar clientes: {str(e)}")

        # Ordenar por data de criação descendente (mais recentes primeiro)
        clients_sorted = sorted(clients, key=lambda c: c.created_at, reverse=True)
        return [self._client_to_response(client) for client in clients_sorted]

    def delete_client(self, client_id: int) -> bool:
        """Remove cliente e seus endereços"""
        client = self.db.query(Client).filter(Client.id == client_id).first()
        if not client:
            return False

        self.db.delete(client)
        self.db.commit()
        return True

    def toggle_client_status(self, client_id: int) -> Client:
        """Ativa/desativa cliente"""
        client = self.db.query(Client).filter(Client.id == client_id).first()
        if not client:
            raise ValueError("Cliente não encontrado")

        return self._client_to_response(client)

    def create_client_from_extracted_data(self, extracted_data: dict) -> ClientResponse:
        """
        Cria cliente a partir de dados extraídos de documento.
        Gera embedding vetorial para busca semântica.
        Lança exceções se houver problemas na criação.
        """
        # Gerar email simples se não houver um válido
        email = extracted_data.get('email')
        if not email:
            # Usar nome simplificado para gerar email único
            name_simple = extracted_data['name'].lower().replace(' ', '.')
            # Remover caracteres não alfanuméricos exceto ponto
            import re
            name_simple = re.sub(r'[^a-z0-9.]', '', name_simple)
            email = f"{name_simple}@temp.document"

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
        client = self.create_client(client_data)

        # Gerar embedding vetorial para busca semântica
        try:
            document_parser = DocumentParserService(self.ai_service)
            embedding = document_parser.generate_client_embedding(extracted_data)

            if embedding:
                # Atualizar cliente com embedding
                db_client = self.db.query(Client).filter(Client.id == client.id).first()
                if db_client:
                    db_client.embedding = embedding
                    db_client.processed = True  # Marcar como processado
                    self.db.commit()
                    self.db.refresh(db_client)
        except Exception as e:
            # Log do erro mas não falha a criação do cliente
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to generate embedding for client {client.id}: {str(e)}")

        return client

    def process_client(self, client_id: int) -> Client:
        """Processa cliente com IA (atualmente apenas marca como processado)"""
        client = self.db.query(Client).filter(Client.id == client_id).first()
        if not client:
            raise ValueError("Cliente não encontrado")

        # Por enquanto, apenas marca como processado
        # Futuramente pode incluir processamento de IA específico para clientes
        client.processed = True
        self.db.commit()
        self.db.refresh(client)

        return self._client_to_response(client)

    def search_similar_clients(self, query: str, limit: int = 10) -> List[Dict]:
      """Busca clientes por similaridade semântica PURA (só embedding CLIP)"""
      try:
          # 1. Gera embedding CLIP do texto
          query_embedding = self.ai_service.generate_clip_text_embedding(query)
          if not query_embedding or len(query_embedding) != 512:
              logger.warning("Embedding inválido gerado")
              return []

          # 2. Query correta (sem SELECT duplicado + vetor como lista)
          sql = text("""
              SELECT
                  id, name, email, cpf, phone,
                  embedding <=> :query_vec AS distance
              FROM clients
              WHERE embedding IS NOT NULL
                AND is_active = true
              ORDER BY embedding <=> :query_vec
              LIMIT :limit
          """)

          results = self.db.execute(sql, {
              "query_vec": f"[{','.join(map(str, query_embedding))}]",  # ← STRING no formato pgvector
              "limit": limit
          }).fetchall()

          similar_clients = []
          for row in results:
              similarity = round((1 - row.distance) * 100, 2)

              # Ajuste esse threshold conforme seus testes (20~35 costuma ser bom)
              if similarity < 15:
                  continue

              client = self.get_client(row.id)  # já carrega addresses
              if client:
                  similar_clients.append({
                      "client_id": row.id,
                      "name": row.name or "Sem nome",
                      "email": row.email,
                      "cpf": row.cpf,
                      "similarity_score": similarity,
                      "client_data": client
                  })

          logger.info(f"Busca: '{query}' → {len(similar_clients)} resultados (threshold 25%)")
          return similar_clients

      except Exception as e:
          logger.error(f"Erro fatal na busca semântica: {e}")
          import traceback
          traceback.print_exc()
          return []
