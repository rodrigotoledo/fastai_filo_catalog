import numpy as np
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import logging

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        # Usar CLIP para processamento real de imagens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Usando dispositivo: {self.device}")

        try:
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model.to(self.device)
            self.embedding_dim = 512  # CLIP ViT-B/32 tem 512 dimensões
            logger.info("CLIP model carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar CLIP model: {e}")
            # Fallback para embeddings simples
            self.model = None
            self.processor = None
            self.embedding_dim = 768

    def process_image(self, image_path: str, user_prompt: str = None) -> Tuple[List[float], str]:
        """
        Processa uma imagem usando CLIP e retorna embedding + descrição
        """
        try:
            if self.model is None:
                # Fallback para método simples
                filename = Path(image_path).name
                embedding = self._simple_embedding()
                description = f"Imagem processada (fallback): {filename}"
                if user_prompt:
                    description += f" | Contexto: {user_prompt}"
                return embedding, description

            # Carregar e processar imagem
            image = Image.open(image_path).convert('RGB')

            # Se temos prompt do usuário, combinar com processamento da imagem
            if user_prompt and user_prompt.strip():
                # Estratégia: Processar imagem + texto juntos
                combined_text = f"Esta é uma foto de {user_prompt}"

                # Processar imagem
                image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                image_features = self.model.get_image_features(**image_inputs)

                # Processar texto combinado
                text_inputs = self.processor(text=[combined_text], return_tensors="pt", padding=True).to(self.device)
                text_features = self.model.get_text_features(**text_inputs)

                # Estratégia adaptativa: se o user_prompt contém palavras-chave específicas,
                # dar mais peso ao texto para melhorar matching exato
                keywords = ['gato', 'cachorro', 'pássaro', 'ave', 'carro', 'praia', 'montanha']
                has_specific_keywords = any(keyword in user_prompt.lower() for keyword in keywords)

                if has_specific_keywords:
                    # Dar muito mais peso ao texto para matching exato
                    combined_embedding = 0.1 * image_features + 0.9 * text_features
                else:
                    # Peso balanceado para descrições gerais
                    combined_embedding = 0.5 * image_features + 0.5 * text_features

                # Normalizar
                combined_embedding = combined_embedding / combined_embedding.norm(dim=-1, keepdim=True)

                embedding = combined_embedding.detach().cpu().numpy().flatten().tolist()
                description = f"Imagem processada com contexto: {user_prompt}"

            else:
                # Processamento normal da imagem
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)

                embedding = image_features.detach().cpu().numpy().flatten().tolist()
                filename = Path(image_path).name
                description = f"Imagem processada com IA: {filename}"

            # Garantir que embedding seja lista de floats
            embedding = [float(x) for x in embedding]

            logger.info(f"Imagem {Path(image_path).name} processada com sucesso")
            return embedding, description

        except Exception as e:
            logger.error(f"Erro ao processar imagem {image_path}: {str(e)}")
            # Fallback
            filename = Path(image_path).name
            return self._simple_embedding(), f"Erro no processamento: {filename}"

    def _simple_embedding(self) -> List[float]:
        """Embedding simples para fallback"""
        np.random.seed(42)  # Para consistência
        return np.random.normal(0, 1, self.embedding_dim).tolist()

    def search_similar_images(self, query_embedding: List[float], embeddings: List[Tuple[int, List[float]]], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Busca imagens similares baseado no embedding
        """
        query_vec = np.array(query_embedding)

        similarities = []
        for photo_id, embedding in embeddings:
            if embedding is not None and len(embedding) > 0:
                try:
                    emb_vec = np.array(embedding)
                    # Similaridade cosseno
                    norm_product = np.linalg.norm(query_vec) * np.linalg.norm(emb_vec)
                    if norm_product > 0:
                        similarity = np.dot(query_vec, emb_vec) / norm_product
                        similarities.append((photo_id, float(similarity)))
                except:
                    continue

        # Ordenar por similaridade (maior primeiro)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def text_to_embedding(self, text: str) -> List[float]:
        """
        Converte texto para embedding usando CLIP
        """
        try:
            if self.model is None:
                # Fallback para método simples
                return self._simple_embedding()

            # Processar texto com CLIP
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)

            # Normalizar e converter para lista
            embedding = text_features.detach().cpu().numpy().flatten().tolist()
            embedding = [float(x) for x in embedding]

            logger.info(f"Texto processado: '{text}'")
            return embedding

        except Exception as e:
            logger.error(f"Erro ao processar texto '{text}': {str(e)}")
            return self._simple_embedding()

    def find_similar_by_text(self, query_text: str, embeddings: List[Tuple[int, List[float]]], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Busca imagens por texto
        """
        query_embedding = self.text_to_embedding(query_text)
        return self.search_similar_images(query_embedding, embeddings, top_k)

    def process_client_text(self, client_data: dict, user_description: str = None) -> Tuple[List[float], str]:
        """
        Processa dados de cliente e gera embedding + descrição usando CLIP
        """
        try:
            # Criar texto descritivo conciso do cliente (limitar tamanho para evitar truncamento)
            text_parts = []

            # Informações principais (sempre incluir)
            if client_data.get('name'):
                text_parts.append(client_data['name'])

            # Adicionar apenas cidade/estado do primeiro endereço (mais útil para busca)
            addresses = client_data.get('addresses', [])
            if addresses and addresses[0].get('city'):
                addr = addresses[0]
                location = f"{addr.get('city', '')}, {addr.get('state', '')}".strip(", ")
                if location:
                    text_parts.append(location)

            # Adicionar nickname se existir (útil para busca informal)
            if client_data.get('nickname'):
                text_parts.append(client_data['nickname'])

            # Combinar com descrição do usuário se fornecida
            base_text = " ".join(text_parts)

            if user_description and user_description.strip():
                # Limitar descrição do usuário para não exceder limite de tokens
                user_desc_short = user_description[:100]  # Limitar a 100 caracteres
                combined_text = f"{base_text} {user_desc_short}".strip()
            else:
                combined_text = base_text

            # Garantir que não está vazio
            if not combined_text.strip():
                combined_text = "Cliente sem informações"

            # Limitar tamanho total para evitar problemas com CLIP
            if len(combined_text) > 200:
                combined_text = combined_text[:200] + "..."

            # Gerar embedding usando CLIP
            embedding = self.text_to_embedding(combined_text)

            # Criar descrição da IA
            ai_description = f"Cliente processado com IA. Dados: {base_text}"
            if user_description:
                ai_description += f" | Contexto adicional: {user_description}"

            logger.info(f"Cliente '{client_data.get('name', 'Unknown')}' processado com sucesso")
            return embedding, ai_description

        except Exception as e:
            logger.error(f"Erro ao processar cliente {client_data.get('name', 'Unknown')}: {str(e)}")
            # Fallback
            return self._simple_embedding(), f"Erro no processamento do cliente: {client_data.get('name', 'Unknown')}"

    def find_similar_clients(self, query_text: str, embeddings: List[Tuple[int, List[float]]], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Busca clientes similares por texto
        """
        query_embedding = self.text_to_embedding(query_text)
        return self.search_similar_images(query_embedding, embeddings, top_k)

    def process_custom_extraction(self, text_content: str, instructions: str) -> str:
        """
        Processa extração de dados customizada com texto e instruções separadas.

        Args:
            text_content: Texto do documento a ser analisado
            instructions: Instruções sobre o que extrair

        Returns:
            Dados extraídos em formato JSON
        """
        try:
            return self._extract_client_data_from_text_and_instructions(text_content, instructions)
        except Exception as e:
            logger.error(f"Erro na extração customizada: {str(e)}")
            return self._extract_client_data_from_text_and_instructions(text_content, "Extraia informações básicas de cliente")
        """
        Processa um prompt customizado usando CLIP para gerar resposta baseada em texto.
        Este método é usado para tarefas como extração de dados estruturados de texto.

        Args:
            prompt: Prompt customizado para processamento

        Returns:
            Resposta gerada baseada no prompt
        """
        try:
            # Para tarefas de extração de dados, vamos usar uma abordagem baseada em templates
            # CLIP não é otimizado para geração de texto, então usaremos uma abordagem simplificada

            # Analisar o prompt para determinar o tipo de tarefa
            prompt_lower = prompt.lower()
            if any(word in prompt_lower for word in ['extraia', 'extrair', 'extract', 'procure', 'busque']):
                # Tarefa de extração de dados de cliente
                return self._extract_client_data_from_custom_prompt(prompt)
            else:
                # Para outros tipos de prompts, retornar uma resposta básica
                return "Prompt processado, mas funcionalidade não implementada para este tipo de tarefa"

        except Exception as e:
            logger.error(f"Erro ao processar prompt customizado: {str(e)}")
            return f"Erro no processamento: {str(e)}"

    def _extract_client_data_from_text_and_instructions(self, text_content: str, instructions: str) -> str:
        """
        Extrai dados de cliente do texto usando instruções específicas.
        """
        import re
        import json

        extracted_data = {
            "name": None,
            "cpf": None,
            "email": None,
            "phone": None,
            "date_of_birth": None,
            "address": {
                "street": None,
                "number": None,
                "complement": None,
                "neighborhood": None,
                "city": None,
                "state": None,
                "postal_code": None
            },
            "notes": None,
            "confidence": "low"
        }

        instructions_lower = instructions.lower()

        # DETECÇÃO ESPECÍFICA PARA DOCUMENTOS BRASILEIROS
        is_brazilian_document = any(word in instructions_lower for word in ['carteira', 'identidade', 'rg', 'cpf', 'brasil', 'brazil'])
        is_brazilian_document = is_brazilian_document or any(word in text_content.lower() for word in ['carteira de identidade', 'registro geral', 'policia civil', 'minas gerais', 'são paulo', 'rio de janeiro'])

        if is_brazilian_document:
            print(f"Documento brasileiro detectado. Aplicando extração específica...")

            # Padrões específicos para RG brasileiro
            # Nome após "Nome / Name"
            name_match = re.search(r'Nome\s*/\s*Name\s*\n([^\n]+)', text_content, re.IGNORECASE)
            if name_match:
                name = name_match.group(1).strip()
                # Filtrar nomes válidos (não headers, não estados)
                if (len(name.split()) >= 2 and
                    len(name) > 5 and
                    not any(word in name.upper() for word in ['ESTADO', 'POLICIA', 'CIVIL', 'CARTEIRA', 'IDENTIDADE', 'SOCIAL'])):
                    extracted_data["name"] = name
                    print(f"Nome extraído: {name}")

            # CPF após "Registro Geral - CPF" ou variações
            cpf_patterns = [
                r'Registro\s+Geral\s*-\s*CPF\s*/\s*Personal\s+Number\s*\n([^\n\d]*(\d{3}[\.\s]?\d{3}[\.\s]?\d{3}[\-\s]?\d{2}))',
                r'CPF\s*/\s*Personal\s+Number\s*\n([^\n\d]*(\d{3}[\.\s]?\d{3}[\.\s]?\d{3}[\-\s]?\d{2}))',
                r'Registro\s+Geral\s*-\s*CPE\s*/\s*Personal\s+Number\s*\n([^\n\d]*(\d{3}[\.\s]?\d{3}[\.\s]?\d{3}[\-\s]?\d{2}))',
                r'(\d{3}[\.\s]?\d{3}[\.\s]?\d{3}[\-\s]?\d{2})'
            ]

            for pattern in cpf_patterns:
                match = re.search(pattern, text_content, re.IGNORECASE)
                if match:
                    # Pegar o último grupo (que contém apenas os dígitos)
                    cpf = match.group(len(match.groups())) if match.groups() else match.group(1)
                    cpf = re.sub(r'[^\d]', '', cpf)  # Limpar formatação
                    if len(cpf) == 11:
                        cpf_formatted = f"{cpf[:3]}.{cpf[3:6]}.{cpf[6:9]}-{cpf[9:]}"
                        extracted_data["cpf"] = cpf_formatted
                        print(f"CPF extraído: {cpf_formatted}")
                        break

            # Data de nascimento
            birth_patterns = [
                r'Data\s+de\s+Nascimento\s*/\s*Date\s+of\s+Birth\s*\n([^\n]+)',
                r'Data\s+de\s+nascimento\s*:\s*(\d{2}/\d{2}/\d{4})',
                r'(\d{2}/\d{2}/\d{4})'
            ]

            for pattern in birth_patterns:
                match = re.search(pattern, text_content, re.IGNORECASE)
                if match:
                    date_str = match.group(1) if len(match.groups()) > 0 else match.group(0)
                    # Validar formato brasileiro DD/MM/YYYY
                    if re.match(r'\d{2}/\d{2}/\d{4}', date_str.strip()):
                        extracted_data["date_of_birth"] = date_str.strip()
                        print(f"Data nascimento: {date_str.strip()}")
                        break

            # Naturalidade/Local de nascimento
            naturalidade_patterns = [
                r'Naturalidade\s*/\s*Place\s+of\s+Birth\s*\n([^\n]+)',
                r'Naturalidade\s*:\s*([^\n]+)',
            ]

            for pattern in naturalidade_patterns:
                match = re.search(pattern, text_content, re.IGNORECASE)
                if match:
                    naturalidade = match.group(1).strip()
                    # Separar cidade/estado se houver
                    if '/' in naturalidade:
                        city, state = naturalidade.split('/', 1)
                        extracted_data["address"]["city"] = city.strip()
                        extracted_data["address"]["state"] = state.strip()
                    else:
                        extracted_data["address"]["city"] = naturalidade
                    print(f"Naturalidade: {naturalidade}")
                    break

            # Se conseguimos extrair dados específicos brasileiros, marcar como alta confiança
            if extracted_data["name"] or extracted_data["cpf"]:
                extracted_data["confidence"] = "high"
                extracted_data["notes"] = f"Documento brasileiro identificado. Extração específica aplicada."
                return json.dumps(extracted_data, ensure_ascii=False, indent=2)

        # Analisar instruções para determinar o que extrair (código original continua)
        should_extract_name = any(word in instructions_lower for word in ['nome', 'name', 'pessoa', 'cliente', 'currículo', 'resume', 'cv'])
        should_extract_email = any(word in instructions_lower for word in ['email', 'e-mail'])
        should_extract_phone = any(word in instructions_lower for word in ['telefone', 'phone', 'celular', 'whatsapp'])
        should_extract_location = any(word in instructions_lower for word in ['localização', 'location', 'endereço', 'address', 'cidade', 'city', 'estado', 'state'])

        # Se nenhuma informação específica foi solicitada, extrair tudo
        if not any([should_extract_name, should_extract_email, should_extract_phone, should_extract_location]):
            should_extract_name = should_extract_email = should_extract_phone = should_extract_location = True

        # Extrair nome se solicitado
        if should_extract_name:
            name_patterns = [
                r'^([A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+)+)(?:\n|$)',
                r'[Nn]ome[:\s]*([A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+)+)',
                r'([A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+)+)(?:\n[A-Z]|$)',
            ]

            for pattern in name_patterns:
                matches = re.findall(pattern, text_content, re.MULTILINE)
                for match in matches:
                    potential_name = match.strip()
                    potential_name = re.sub(r'\s+', ' ', potential_name)

                    # Limpar títulos profissionais comuns
                    professional_titles = [
                        'senior software developer', 'software developer', 'software engineer',
                        'developer', 'engineer', 'analyst', 'consultant', 'manager', 'director',
                        'architect', 'specialist', 'coordinator', 'technician', 'administrator',
                        'designer', 'scientist', 'lead', 'principal', 'junior', 'pleno'
                    ]

                    name_lower = potential_name.lower()
                    for title in professional_titles:
                        if title in name_lower:
                            potential_name = re.sub(r'\b' + re.escape(title) + r'\b', '', potential_name, flags=re.IGNORECASE).strip()

                    # Verificar se é um nome válido
                    if (len(potential_name.split()) >= 2 and
                        len(potential_name) > 3 and len(potential_name) < 50 and
                        not re.search(r'\d', potential_name) and
                        not any(word in potential_name.lower() for word in ['contact', 'phone', 'email', 'address'])):
                        extracted_data["name"] = potential_name
                        break
                if extracted_data["name"]:
                    break

        # Extrair email se solicitado
        if should_extract_email:
            email_pattern = r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
            match = re.search(email_pattern, text_content)
            if match:
                extracted_data["email"] = match.group(1)

        # Extrair telefone se solicitado
        if should_extract_phone:
            phone_patterns = [
                r'[Tt]elefone?[:\s]*(\(?\d{2}\)?\s*\d{4,5}-?\d{4})',
                r'[Ff]one[:\s]*(\(?\d{2}\)?\s*\d{4,5}-?\d{4})',
                r'[Ww]hats[A-Z]*[:\s]*(\(?\d{2}\)?\s*\d{4,5}-?\d{4})',
                r'\+?55\s*(\(?\d{2}\)?\s*\d{4,5}-?\d{4})',  # Padrão brasileiro com +55
                r'(\(?\d{2}\)?\s*\d{4,5}-?\d{4})',
            ]

            for pattern in phone_patterns:
                match = re.search(pattern, text_content, re.IGNORECASE)
                if match:
                    phone = match.group(1) if len(match.groups()) > 0 else match.group(0)
                    extracted_data["phone"] = phone.strip()
                    break

        # Extrair localização se solicitado
        if should_extract_location:
            # Cidade e estado
            location_patterns = [
                r'[Cc]idade[:\s]*([^\n,]+)',
                r'[Ll]ocalização[:\s]*([^\n,]+)',
                r'([A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+)*),\s*([A-Z]{2})',
                r'([A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+)*)\s*-\s*([A-Z]{2})',
            ]

            for pattern in location_patterns:
                match = re.search(pattern, text_content, re.IGNORECASE)
                if match:
                    if len(match.groups()) >= 2 and match.group(2):
                        # Padrão "Cidade, UF" ou "Cidade - UF"
                        extracted_data["address"]["city"] = match.group(1).strip()
                        extracted_data["address"]["state"] = match.group(2).strip()
                    else:
                        # Apenas cidade
                        extracted_data["address"]["city"] = match.group(1).strip()
                    break

        # Calcular confiança
        filled_fields = sum(1 for value in extracted_data.values()
                          if value is not None and value != "" and value != {})
        if isinstance(extracted_data.get("address"), dict):
            filled_fields += sum(1 for value in extracted_data["address"].values()
                               if value is not None and value != "")

        if filled_fields >= 3:
            extracted_data["confidence"] = "high"
        elif filled_fields >= 2:
            extracted_data["confidence"] = "medium"

        # Adicionar notas
        extracted_data["notes"] = f"Análise baseada nas instruções: {instructions[:100]}..."

        return json.dumps(extracted_data, ensure_ascii=False, indent=2)
        """
        Extrai dados de cliente usando um prompt personalizado fornecido pelo usuário.
        O prompt deve conter instruções sobre o que extrair e o texto a ser analisado.
        """
        import re
        import json

        extracted_data = {
            "name": None,
            "cpf": None,
            "email": None,
            "phone": None,
            "date_of_birth": None,
            "address": {
                "street": None,
                "number": None,
                "complement": None,
                "neighborhood": None,
                "city": None,
                "state": None,
                "postal_code": None
            },
            "notes": None,
            "confidence": "low"
        }

        # Separar o prompt das instruções do texto a ser analisado
        # Procurar por marcadores comuns
        text_content = ""
        instructions = prompt

        # Procurar por "Text content:" ou similar
        text_markers = ["text content:", "texto:", "conteúdo:", "documento:"]
        for marker in text_markers:
            if marker in prompt.lower():
                parts = prompt.lower().split(marker, 1)
                if len(parts) > 1:
                    instructions = parts[0].strip()
                    text_content = parts[1].strip()
                    break

        # Se não encontrou marcador, assumir que tudo é texto a ser analisado
        if not text_content:
            text_content = prompt

        # Analisar as instruções para determinar o que extrair
        instructions_lower = instructions.lower()

        # Sempre tentar extrair informações básicas se mencionadas ou se for um currículo/cliente
        should_extract_name = any(word in instructions_lower for word in ['nome', 'name', 'pessoa', 'currículo', 'resume', 'cv'])
        should_extract_email = any(word in instructions_lower for word in ['email', 'e-mail'])
        should_extract_phone = any(word in instructions_lower for word in ['telefone', 'phone', 'celular', 'whatsapp'])
        should_extract_location = any(word in instructions_lower for word in ['localização', 'location', 'endereço', 'address', 'cidade', 'city'])

        # Se nenhuma informação específica foi solicitada, extrair tudo
        if not any([should_extract_name, should_extract_email, should_extract_phone, should_extract_location]):
            should_extract_name = should_extract_email = should_extract_phone = should_extract_location = True

        # Extrair nome se solicitado
        if should_extract_name:
            name_patterns = [
                r'[Nn]ome[:\s]*([^:\n]+)',
                r'^([A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+)+)(?:\n|$)',
                r'([A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+)+)(?:\n[A-Z]|$)',
            ]

            for pattern in name_patterns:
                matches = re.findall(pattern, text_content, re.MULTILINE)
                for match in matches:
                    potential_name = match.strip()
                    potential_name = re.sub(r'\s+', ' ', potential_name)

                    # Limpar títulos profissionais
                    professional_titles = [
                        'senior software developer', 'software developer', 'developer', 'engineer',
                        'analyst', 'consultant', 'manager', 'director', 'architect', 'specialist'
                    ]
                    name_lower = potential_name.lower()
                    for title in professional_titles:
                        if title in name_lower:
                            potential_name = re.sub(r'\b' + re.escape(title) + r'\b', '', potential_name, flags=re.IGNORECASE).strip()

                    if (len(potential_name.split()) >= 2 and
                        len(potential_name) > 3 and len(potential_name) < 50 and
                        not re.search(r'\d', potential_name)):
                        extracted_data["name"] = potential_name
                        extracted_data["confidence"] = "medium"
                        break
                if extracted_data["name"]:
                    break

        # Extrair email se solicitado
        if should_extract_email:
            email_pattern = r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
            match = re.search(email_pattern, text_content)
            if match:
                extracted_data["email"] = match.group(1)

        # Extrair telefone se solicitado
        if should_extract_phone:
            phone_patterns = [
                r'[Tt]elefone?[:\s]*(\(?\d{2}\)?\s*\d{4,5}-?\d{4})',
                r'[Ff]one[:\s]*(\(?\d{2}\)?\s*\d{4,5}-?\d{4})',
                r'[Ww]hats[A-Z]*[:\s]*(\(?\d{2}\)?\s*\d{4,5}-?\d{4})',
                r'(\(?\d{2}\)?\s*\d{4,5}-?\d{4})',
            ]

            for pattern in phone_patterns:
                match = re.search(pattern, text_content, re.IGNORECASE)
                if match:
                    extracted_data["phone"] = match.group(1).strip()
                    break

        # Extrair localização se solicitado
        if should_extract_location:
            # Cidade e estado
            city_patterns = [
                r'[Cc]idade[:\s]*([^\n,]+)',
                r'([A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+)*),\s*([A-Z]{2})',
            ]

            for pattern in city_patterns:
                match = re.search(pattern, text_content, re.IGNORECASE)
                if match:
                    if ',' in match.group(0):  # Padrão "Cidade, UF"
                        city_state = match.group(0).split(',')
                        extracted_data["address"]["city"] = city_state[0].strip()
                        if len(city_state) > 1:
                            extracted_data["address"]["state"] = city_state[1].strip()
                    else:
                        extracted_data["address"]["city"] = match.group(1).strip()
                    break

        # Calcular confiança baseada no que foi extraído
        filled_fields = sum(1 for value in extracted_data.values()
                          if value is not None and value != "" and value != {})
        if isinstance(extracted_data.get("address"), dict):
            filled_fields += sum(1 for value in extracted_data["address"].values()
                               if value is not None and value != "")

        if filled_fields >= 3:
            extracted_data["confidence"] = "high"
        elif filled_fields >= 2:
            extracted_data["confidence"] = "medium"

        # Adicionar notas
        extracted_data["notes"] = f"Extraído baseado nas instruções: {instructions[:100]}..."

        return json.dumps(extracted_data, ensure_ascii=False, indent=2)
        """
        Extrai dados de cliente de um prompt usando análise de texto básica.
        Esta é uma implementação simplificada - em produção, poderia usar um modelo de linguagem mais avançado.
        """
        import re
        import json

        extracted_data = {
            "name": None,
            "cpf": None,
            "email": None,
            "phone": None,
            "date_of_birth": None,
            "address": {
                "street": None,
                "number": None,
                "complement": None,
                "neighborhood": None,
                "city": None,
                "state": None,
                "postal_code": None
            },
            "notes": None,
            "confidence": "low"
        }

        # Limpar o prompt - remover a parte do prompt de instrução e focar no conteúdo real
        # Procurar pela seção "Text content:" e extrair apenas o conteúdo do documento
        text_content_match = re.search(r'Text content:\s*(.+)', prompt, re.DOTALL | re.IGNORECASE)
        if text_content_match:
            actual_content = text_content_match.group(1).strip()
            # Remover aspas e limitadores se houver
            actual_content = re.sub(r'^["\']|["\']$', '', actual_content)
            # Usar apenas o conteúdo real para extração
            content_to_analyze = actual_content
        else:
            content_to_analyze = prompt

        # Procurar por nome (padrões comuns em português)
        # Estratégia melhorada para currículos e documentos estruturados
        name_patterns = [
            r'[Nn]ome[:\s]*([^:\n]+)',
            r'[Cc]liente[:\s]*([^:\n]+)',
            # Padrão para currículos - nome geralmente aparece no topo
            r'^([A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+)+)(?:\n|$)',
            # Nome seguido de título profissional
            r'^([A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+)+)\n[A-Z][a-z]+',
            # Nome isolado em linha
            r'^([A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+)+)$',
        ]

        for pattern in name_patterns:
            matches = re.findall(pattern, content_to_analyze, re.MULTILINE)
            for match in matches:
                potential_name = match.strip()
                # Limpar nome - remover títulos profissionais comuns
                professional_titles = [
                    'senior software developer', 'software developer', 'developer', 'engineer',
                    'analyst', 'consultant', 'manager', 'director', 'architect', 'specialist',
                    'coordinator', 'technician', 'administrator', 'designer', 'scientist'
                ]
                name_lower = potential_name.lower()
                for title in professional_titles:
                    if title in name_lower:
                        potential_name = potential_name.replace(title, '').strip()
                        break
                # Verificar se parece um nome válido (pelo menos duas palavras, não contém palavras do prompt)
                if (len(potential_name.split()) >= 2 and
                    len(potential_name) > 3 and len(potential_name) < 50 and  # Nome não muito longo
                    not any(word in potential_name.upper() for word in ['ANALYZE', 'EXTRACT', 'CLIENT', 'INFORMATION', 'TEXT', 'CONTENT', 'RESUME', 'CURRICULO', 'CV']) and
                    not re.search(r'\d', potential_name)):  # Não deve conter números
                    extracted_data["name"] = potential_name
                    extracted_data["confidence"] = "medium"
                    break
            if extracted_data["name"]:
                break

        # Procurar por CPF
        cpf_patterns = [
            r'[Cc][Pp][Ff][:\s]*(\d{3}\.?\d{3}\.?\d{3}-?\d{2})',
            r'(\d{3}\.?\d{3}\.?\d{3}-?\d{2})',
        ]

        for pattern in cpf_patterns:
            match = re.search(pattern, content_to_analyze, re.IGNORECASE)
            if match:
                extracted_data["cpf"] = match.group(1)
                break

        # Procurar por email
        email_pattern = r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        match = re.search(email_pattern, content_to_analyze)
        if match:
            extracted_data["email"] = match.group(1)

        # Procurar por telefone
        phone_patterns = [
            r'[Tt]elefone?[:\s]*(\(?\d{2}\)?\s*\d{4,5}-?\d{4})',
            r'[Ff]one[:\s]*(\(?\d{2}\)?\s*\d{4,5}-?\d{4})',
            r'(\(?\d{2}\)?\s*\d{4,5}-?\d{4})',
        ]

        for pattern in phone_patterns:
            match = re.search(pattern, content_to_analyze, re.IGNORECASE)
            if match:
                extracted_data["phone"] = match.group(1).strip()
                break

        # Procurar por data de nascimento
        dob_patterns = [
            r'[Dd]ata.*[Nn]ascimento[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
            r'[Nn]ascimento[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})',  # Data geral
        ]

        for pattern in dob_patterns:
            match = re.search(pattern, content_to_analyze, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                # Converter para formato YYYY-MM-DD
                try:
                    if '/' in date_str:
                        day, month, year = date_str.split('/')
                    elif '-' in date_str:
                        day, month, year = date_str.split('-')
                    else:
                        continue

                    if len(year) == 2:
                        year = f"20{year}" if int(year) < 50 else f"19{year}"

                    extracted_data["date_of_birth"] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                except:
                    pass
                break

        # Procurar por endereço
        # Rua/Logradouro
        street_patterns = [
            r'[Rr]ua[:\s]*([^\n,]+?)(?:\s*,\s*n[º°]?[:\s]*(\d+))?',
            r'[Ee]ndereço[:\s]*([^\n,]+?)(?:\s*,\s*n[º°]?[:\s]*(\d+))?',
        ]

        for pattern in street_patterns:
            match = re.search(pattern, content_to_analyze, re.IGNORECASE)
            if match:
                extracted_data["address"]["street"] = match.group(1).strip()
                if len(match.groups()) > 1 and match.group(2):
                    extracted_data["address"]["number"] = match.group(2)
                break

        # Cidade
        city_pattern = r'[Cc]idade[:\s]*([^\n,]+)'
        match = re.search(city_pattern, content_to_analyze, re.IGNORECASE)
        if match:
            extracted_data["address"]["city"] = match.group(1).strip()

        # Estado
        state_pattern = r'[Ee]stado[:\s]*([A-Z]{2})'
        match = re.search(state_pattern, content_to_analyze, re.IGNORECASE)
        if match:
            extracted_data["address"]["state"] = match.group(1)

        # CEP
        cep_pattern = r'[Cc][Ee][Pp][:\s]*(\d{5}-?\d{3})'
        match = re.search(cep_pattern, content_to_analyze, re.IGNORECASE)
        if match:
            extracted_data["address"]["postal_code"] = match.group(1)

        # Aumentar confiança se encontramos múltiplos campos
        filled_fields = sum(1 for value in extracted_data.values()
                          if value is not None and value != "" and value != {})
        if isinstance(extracted_data.get("address"), dict):
            filled_fields += sum(1 for value in extracted_data["address"].values()
                               if value is not None and value != "")

        if filled_fields >= 3:
            extracted_data["confidence"] = "high"
        elif filled_fields >= 2:
            extracted_data["confidence"] = "medium"

        # Adicionar notas com texto restante relevante (limitar tamanho)
        extracted_data["notes"] = content_to_analyze[:200] + "..." if len(content_to_analyze) > 200 else content_to_analyze

        return json.dumps(extracted_data, ensure_ascii=False, indent=2)
