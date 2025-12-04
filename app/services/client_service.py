from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_
from typing import List, Optional
from app.models.client import Client, ClientAddress
from app.schemas.client import (
    ClientCreate, ClientUpdate, ClientResponse,
    ClientAddressCreate, ClientAddressResponse, ClientDocuments
)
import re
from datetime import datetime

class ClientService:
    def __init__(self, db: Session):
        self.db = db

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

        client.is_active = not client.is_active
        self.db.commit()
        self.db.refresh(client)

        return self._client_to_response(client)
