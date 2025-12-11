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
from langchain.prompts import PromptTemplate
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

        total_found = (total + page_size - 1) // page_size

        return {
            "results": [self._client_to_response(client) for client in clients],
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_found": total_found,
            "has_next": page < total_found,
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

    def build_client_text_for_embedding(self, client):
        lines = ["Cliente que possui as informações abaixo (dados pessoais, contatos e endereços):", ""]

        # === NOME - várias formas ===
        nome_completo = (client.name or "").strip()
        primeiro_nome = nome_completo.split()[0] if nome_completo else ""
        ultimo_nome = " ".join(nome_completo.split()[-1:]) if nome_completo else "" if len(nome_completo.split()) > 1 else nome_completo

        lines.extend([
            f"Nome completo: {nome_completo}",
            f"Primeiro nome: {primeiro_nome}",
            f"Último nome: {ultimo_nome}",
            f"Apelido: {client.nickname or ''}",
            "",
        ])

        # === E-MAIL - quebrado ao extremo ===
        email = (client.email or "").strip().lower()
        if email and "@" in email:
            usuario_email, dominio_completo = email.split("@", 1)
            provedor = dominio_completo.split(".")[0]  # gmail, hotmail, yahoo, etc
        else:
            usuario_email = dominio_completo = provedor = ""

        lines.extend([
            f"E-mail completo: {email}",
            f"Usuário do e-mail: {usuario_email}",
            f"Domínio do e-mail: {dominio_completo}",
            f"Provedor de e-mail: {provedor}",
            "",
        ])

        # === TELEFONE ===
        lines.append(f"Telefone: {client.phone or ''}")
        lines.append("")

        # === DOCUMENTOS ===
        lines.extend([
            f"CPF: {client.cpf or ''}",
            f"RG: {client.rg or ''}",
            "",
        ])

        # === DATA DE NASCIMENTO - em TODOS os formatos possíveis ===
        if client.birth_date:
            if isinstance(client.birth_date, str):
                try:
                    birth = datetime.strptime(client.birth_date, "%Y-%m-%d")
                except:
                    birth = None
            else:
                birth = client.birth_date

            if birth:
                linhas_data = [
                    f"Data de nascimento: {birth.strftime('%d/%m/%Y')}",
                    f"Data de nascimento (dia/mês/ano): {birth.day:02d}/{birth.month:02d}/{birth.year}",
                    f"Data de nascimento (ISO): {birth.strftime('%Y-%m-%d')}",
                    f"Data de nascimento (AAAAMMDD): {birth.strftime('%Y%m%d')}",
                    f"Ano de nascimento: {birth.year}",
                    f"Mês de nascimento: {birth.month:02d}",
                    f"Dia de nascimento: {birth.day:02d}",
                ]
            else:
                linhas_data = ["Data de nascimento: "]
        else:
            linhas_data = ["Data de nascimento: "]

        lines.extend(linhas_data)
        lines.append("")

        # === ENDEREÇOS - também bem detalhados ===
        if client.addresses:
            for i, addr in enumerate(client.addresses, 1):
                lines.append(f"ENDEREÇO {i}")
                lines.extend([
                    f"Rua/Av: {addr.street or ''}",
                    f"Número: {addr.number or ''}",
                    f"Complemento: {addr.complement or ''}",
                    f"Bairro: {addr.neighborhood or ''}",
                    f"Cidade: {addr.city or ''}",
                    f"Estado: {addr.state or ''}",
                    f"UF: {addr.state or ''}",
                    f"CEP: {addr.zip_code or ''}",
                    f"Endereço completo: {', '.join(filter(None, [addr.street, addr.number, addr.complement, addr.neighborhood, addr.city, addr.state, addr.zip_code]))}",
                    "",
                ])
        else:
            lines.append("Endereço: Nenhum endereço cadastrado")
            lines.append("")

        # Limpa linhas vazias no final
        while lines and not lines[-1].strip():
            lines.pop()

        full_text = "\n".join(lines).strip()
        return full_text

    def process_client(self, client_id: int) -> Client:
      """Gera embedding com Gemini e salva direto no campo embedding do Client"""
      client = self.db.query(Client).filter(Client.id == client_id).first()
      if not client:
          raise ValueError("Cliente não encontrado")

      full_text = self.build_client_text_for_embedding(client)

      if not full_text:
          raise ValueError("Cliente sem dados para gerar embedding")

      # === 2. Gera o embedding usando a função que você já tem ===
      embedding_vector = self.ai_service.generate_embedding_complex(full_text)

      # === 3. Salva direto no cliente ===
      client.embedding = embedding_vector          # ← campo já existe no model
      client.processed = True

      self.db.commit()
      self.db.refresh(client)

      return self._client_to_response(client)

    def search_similar_clients(self, query: str, limit: int = 10) -> List[Dict]:
      """Busca clientes por similaridade semântica usando embeddings configuráveis"""
      try:
          import os

          # Detectar dimensões baseadas na configuração
          ai_model_type = os.getenv("AI_MODEL_TYPE", "gemini").lower().strip()
          expected_dims = {
              "gemini": 768,
              "openai": 1536,
              "local": 384
          }.get(ai_model_type, 768)

          # 1. Gera embedding usando a função configurada
          query_embedding = self.ai_service.generate_embedding_complex(query)
          if not query_embedding or len(query_embedding) != expected_dims:
              logger.warning(f"Embedding inválido gerado: {len(query_embedding) if query_embedding else 0} dims, esperado {expected_dims} para {ai_model_type}")
              return []

          # 2. Estima threshold dinâmico baseado na query
          dynamic_threshold = self.estimate_search_threshold(query)
          logger.info(f"Busca semântica: query='{query}', threshold={dynamic_threshold}, limit={limit}")

          # 3. Query otimizada com threshold dinâmico
          sql = text("""
              SELECT
                  clients.*,
                  embedding <=> :query_vec AS distance
              FROM clients
              WHERE embedding IS NOT NULL
                AND is_active = true
                AND (1 - (embedding <=> :query_vec)) >= :threshold
              ORDER BY embedding <=> :query_vec
              LIMIT :limit
          """)

          results = self.db.execute(sql, {
              "query_vec": f"[{','.join(map(str, query_embedding))}]",  # ← STRING no formato pgvector
              "threshold": dynamic_threshold,
              "limit": limit
          }).fetchall()

          # Converter para dicionários com similarity_score
          clients_data = []
          for row in results:
              # row é um Row object do SQLAlchemy, converter para dict corretamente
              client_data = {column: getattr(row, column) for column in row._fields if column != 'distance'}
              distance = getattr(row, 'distance', None)
              client_data['similarity_score'] = 1 - distance if distance is not None else None
              clients_data.append(client_data)

          return clients_data

      except Exception as e:
          logger.error(f"Erro fatal na busca semântica: {e}")
          import traceback
          traceback.print_exc()
          return []

    def estimate_search_threshold(self, query: str) -> float:
        """
        Estima o threshold ideal de similaridade para uma query consultando diretamente o LLM.
        Usa um prompt especializado para determinar o threshold ótimo baseado na análise da query.
        """
        try:
            # Prompt especializado para determinar threshold
            threshold_prompt = f"""
            Eu sou um Agente especializado em procurar um ótimo threshold e baseado na pesquisa a seguir gostaria de um possível numero a usar que buscaria os melhores resultados:

            Query: "{query}"

            Analise esta query de busca e determine o threshold ideal de similaridade (0.0 a 1.0) para busca vetorial em um sistema de busca de clientes. Mas voce precisa usar sua propria experiencia pra me ajudar, preciso tentar colocar o valor que obtermos justamente para realizar outras pesquisas tentando acertar bem, e pelo visto a ideia eh sempre termos valores abaixo de 0.5 e acima de 0.3.

            Retorne apenas um número decimal entre 0.0 e 1.0 representando o threshold recomendado.
            Não inclua explicações, apenas o número.
            """

            # Usar LangChain diretamente para consultar o LLM
            prompt = PromptTemplate(
                template=threshold_prompt,
                input_variables=[]
            )

            chain = prompt | self.ai_service.llm
            result = chain.invoke({})

            # Extrair o conteúdo da resposta
            result_text = result.content if hasattr(result, 'content') else str(result)

            # Extrair número da resposta
            import re
            threshold_match = re.search(r'(\d+\.?\d*)', result_text.strip())
            if threshold_match:
                threshold = float(threshold_match.group(1))
                # Garantir que está entre 0.0 e 1.0
                threshold = max(0.0, min(1.0, threshold))
                logger.info(f"LLM determinou threshold {threshold} para query: '{query}'")
                return threshold

        except Exception as e:
            logger.warning(f"Erro ao consultar LLM sobre threshold: {e}")

        # Fallback para heurísticas se LLM falhar
        logger.info(f"Usando fallback de heurísticas para query: '{query}'")
        return self._estimate_threshold_fallback(query)

    def _estimate_threshold_fallback(self, query: str) -> float:
        """
        Método de fallback usando heurísticas quando a IA não está disponível
        """
        query_lower = query.lower().strip()
        words = query.split()
        word_count = len(words)

        # 1. Queries muito específicas (alta precisão necessária)
        specific_indicators = [
            '@', '.com', '.org', '.net', 'gmail', 'hotmail', 'yahoo',  # emails
            'cpf', 'rg', 'cnpj',  # documentos
            'telefone', 'celular', 'fone',  # contatos
            'rua', 'avenida', 'av.', 'alameda', 'travessa'  # endereços específicos
        ]

        if any(indicator in query_lower for indicator in specific_indicators):
            return 0.75  # Threshold alto para queries específicas

        # 2. Queries com nomes próprios (palavras capitalizadas)
        capitalized_words = [w for w in words if w and w[0].isupper() and len(w) > 1]
        if len(capitalized_words) >= 2:  # Provavelmente nome + sobrenome
            return 0.65

        # 3. Queries numéricas (datas, CEPs, etc.)
        import re
        if re.search(r'\d{4,}', query):  # Pelo menos 4 dígitos consecutivos
            return 0.70

        # 4. Queries curtas e diretas (1-2 palavras)
        if word_count <= 2:
            if word_count == 1:
                return 0.20  # Muito genérico
            else:
                return 0.35  # Duas palavras, moderadamente específico

        # 5. Queries descritivas (3+ palavras)
        if word_count >= 5:
            return 0.45  # Queries longas precisam de threshold mais alto

        # 6. Queries com conectores lógicos
        logical_indicators = [' e ', ' ou ', ' com ', ' de ', ' em ', ' para ', ' que ']
        if any(indicator in query_lower for indicator in logical_indicators):
            return 0.40

        # 7. Default para queries médias
        return 0.35

    def process_all_clients(self, limit: int = None) -> Dict:
        """Processa todos os clientes para gerar embeddings"""
        query = self.db.query(Client)

        if limit:
            query = query.limit(limit)

        unprocessed_clients = query.all()

        if not unprocessed_clients:
            return {"processed": 0, "total": 0, "message": "Nenhum cliente para processar"}

        processed_count = 0
        errors = 0

        for client in unprocessed_clients:
            try:
                # Usar o mesmo método process_client existente
                self.process_client(client.id)
                processed_count += 1

                # Log de progresso a cada 10 clientes
                if processed_count % 10 == 0:
                    logger.info(f"Processados {processed_count}/{len(unprocessed_clients)} clientes")

            except Exception as e:
                errors += 1
                logger.error(f"Erro ao processar cliente {client.id} ({client.name}): {str(e)}")
                continue

        return {
            "processed": processed_count,
            "errors": errors,
            "total": len(unprocessed_clients),
            "message": f"Processamento concluído: {processed_count} processados, {errors} erros"
        }
