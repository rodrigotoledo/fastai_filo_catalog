# ai_service.py - VERSÃO LANGCHAIN 2025 (IMAGENS + EXTRAÇÃO DE CLIENTES)
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from fastapi import UploadFile
from dotenv import load_dotenv
import json
from langchain_huggingface import HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.messages import HumanMessage
import base64
import mimetypes
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        self.store_id = self._get_or_create_store()
        # Inicializar LangChain com modelo local
        self.llm = self._initialize_langchain_model()
        # Initialize Embeddings
        self.embeddings = self._initialize_embeddings()
        # Initialize CLIP for image embeddings
        self.clip_model, self.clip_processor = self._initialize_clip()
        logger.info(f"AIService iniciado com LangChain | Store ID: {self.store_id}")

    def _initialize_langchain_model(self):
        """Inicializa o modelo LangChain com fallback inteligente"""
        # Ordem de prioridade: OpenAI -> Local -> Anthropic -> Gemini
        providers = [
            ("openai", self._initialize_openai_model),
            ("local", self._initialize_local_model),
            ("anthropic", self._initialize_anthropic_model),
            ("gemini", self._initialize_gemini_model)
        ]

        # Verificar se há um provedor específico configurado
        forced_provider = os.getenv("AI_MODEL_TYPE", "").lower().strip()
        if forced_provider:
            provider_map = {name: func for name, func in providers}
            if forced_provider in provider_map:
                try:
                    llm = provider_map[forced_provider]()
                    if llm:
                        logger.info(f"Modelo forçado '{forced_provider}' inicializado com sucesso")
                        return llm
                except Exception as e:
                    logger.warning(f"Modelo forçado '{forced_provider}' falhou, tentando outros provedores")
                    # Remover o provedor forçado da lista de fallback
                    providers = [(name, func) for name, func in providers if name != forced_provider]

        # Tentar provedores em ordem de prioridade (exceto o forçado que já falhou)
        for provider_name, init_func in providers:
            try:
                llm = init_func()
                if llm:
                    logger.info(f"Modelo '{provider_name}' inicializado com sucesso")
                    return llm
            except Exception as e:
                logger.warning(f"Falha ao inicializar {provider_name}: {str(e)}")
                continue

        logger.error("Nenhum modelo de IA conseguiu inicializar")
        return None

    def _initialize_openai_model(self):
        """Inicializa OpenAI GPT-4"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY não configurada")

            llm = ChatOpenAI(
                model="gpt-4o-mini",  # ou "gpt-4o" para melhor qualidade
                openai_api_key=api_key,
                temperature=0.7,
                max_tokens=512
            )
            logger.info("Modelo OpenAI inicializado via LangChain")
            return llm

        except Exception as e:
            logger.error(f"Erro ao inicializar OpenAI: {str(e)}")
            return None

    def _initialize_anthropic_model(self):
        """Inicializa Anthropic Claude"""
        try:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY não configurada")

            llm = ChatAnthropic(
                model="claude-3-haiku-20240307",  # ou "claude-3-sonnet-20240229"
                anthropic_api_key=api_key,
                temperature=0.7,
                max_tokens=512
            )
            logger.info("Modelo Anthropic inicializado via LangChain")
            return llm

        except Exception as e:
            logger.error(f"Erro ao inicializar Anthropic: {str(e)}")
            return None

    def _initialize_gemini_model(self):
        """Inicializa Google Gemini via LangChain"""
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY não configurada")

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",  # Multimodal model
                google_api_key=api_key,
                temperature=0.4,
                max_tokens=1024
            )
            logger.info("Modelo Gemini inicializado via LangChain")
            return llm

        except Exception as e:
            logger.error(f"Erro ao inicializar Gemini: {str(e)}")
            return None

    def _initialize_local_model(self):
        """Inicializa modelo local via HuggingFace"""
        try:
            # Usar um modelo de texto para geração de descrições
            model_name = os.getenv("LOCAL_MODEL", "microsoft/DialoGPT-medium")

            # Para geração de texto, usar pipeline de text-generation
            pipe = pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256  # GPT-like models
            )

            llm = HuggingFacePipeline(pipeline=pipe)
            logger.info(f"Modelo local LangChain inicializado: {model_name}")
            return llm

        except Exception as e:
            logger.error(f"Erro ao inicializar modelo local: {str(e)}")
            # Fallback para modelo mais simples
            try:
                pipe = pipeline(
                    "text-generation",
                    model="gpt2",
                    max_new_tokens=512,
                    temperature=0.7
                )
                return HuggingFacePipeline(pipeline=pipe)
            except Exception as e2:
                logger.error(f"Fallback também falhou: {str(e2)}")
                return None

    def _get_or_create_store(self) -> str:
        store_id = os.getenv("FILE_SEARCH_STORE_ID", "").strip()
        if store_id and store_id.startswith("file_search_stores/"):
            # Como não temos mais Gemini, apenas retornar None
            logger.warning("FILE_SEARCH_STORE_ID não suportado sem Gemini API")
            return None

        logger.warning("Usando busca simplificada sem File Search Store")
        return None

    def _initialize_embeddings(self):
        """Initialize embeddings based on AI_MODEL_TYPE configuration"""
        try:
            # Verificar qual modelo foi configurado
            ai_model_type = os.getenv("AI_MODEL_TYPE", "gemini").lower().strip()

            if ai_model_type == "openai":
                return self._initialize_openai_embeddings()
            elif ai_model_type == "gemini":
                return self._initialize_gemini_embeddings()
            elif ai_model_type == "local":
                return self._initialize_local_embeddings()
            else:
                logger.warning(f"AI_MODEL_TYPE '{ai_model_type}' não suportado para embeddings. Usando Gemini como fallback.")
                return self._initialize_gemini_embeddings()

        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            return None

    def _initialize_gemini_embeddings(self):
        """Initialize Google Gemini Embeddings"""
        try:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY ou GEMINI_API_KEY não configurada")

            return GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key
            )
        except Exception as e:
            logger.error(f"Erro ao inicializar Gemini embeddings: {str(e)}")
            return None

    def _initialize_openai_embeddings(self):
        """Initialize OpenAI Embeddings"""
        try:
            from langchain_openai import OpenAIEmbeddings

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY não configurada")

            return OpenAIEmbeddings(
                model="text-embedding-3-small",  # 1536 dimensions
                openai_api_key=api_key
            )
        except Exception as e:
            logger.error(f"Erro ao inicializar OpenAI embeddings: {str(e)}")
            return None

    def _initialize_local_embeddings(self):
        """Initialize local embeddings using sentence-transformers"""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            logger.error(f"Erro ao inicializar embeddings locais: {str(e)}")
            return None

    def _initialize_clip(self):
        """Initialize CLIP model for image embeddings"""
        try:
            model_name = "openai/clip-vit-base-patch32"
            model = CLIPModel.from_pretrained(model_name)
            processor = CLIPProcessor.from_pretrained(model_name)
            logger.info(f"CLIP model initialized: {model_name}")
            return model, processor
        except Exception as e:
            logger.error(f"Failed to initialize CLIP: {e}")
            return None, None

    def generate_embedding(self, image_path: str) -> List[float]:
        """Generate embedding vector for image using CLIP"""
        if not self.clip_model or not self.clip_processor:
            logger.warning("CLIP model not initialized. Returning zero vector as fallback.")
            return [0.0] * 512

        try:

            image = Image.open(image_path)
            inputs = self.clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.clip_model.get_image_features(**inputs)
            embedding = outputs.squeeze().tolist()
            # CLIP embeddings are 512 dimensions, perfect
            return embedding
        except Exception as e:
            logger.error(f"Error generating image embedding: {e}")
            return [0.0] * 512


    def generate_embedding_complex(self, text: str) -> List[float]:
        """Generate embedding based on configured AI_MODEL_TYPE"""
        if not self.embeddings:
            # Fallback baseado no tipo configurado
            ai_model_type = os.getenv("AI_MODEL_TYPE", "gemini").lower().strip()
            if ai_model_type == "openai":
                return [0.0] * 1536  # OpenAI text-embedding-3-small
            elif ai_model_type == "local":
                return [0.0] * 384   # sentence-transformers/all-MiniLM-L6-v2
            else:
                return [0.0] * 768   # Gemini default

        try:
            embedding = self.embeddings.embed_query(text)

            # Log das dimensões para debug
            ai_model_type = os.getenv("AI_MODEL_TYPE", "gemini").lower().strip()
            expected_dims = {
                "gemini": 768,
                "openai": 1536,
                "local": 384
            }.get(ai_model_type, 768)

            if len(embedding) != expected_dims:
                logger.warning(f"Embedding size inesperado para {ai_model_type}: {len(embedding)}, esperado {expected_dims}")

            return embedding
        except Exception as e:
            logger.error(f"Erro ao gerar embedding: {e}")
            # Fallback com dimensões corretas
            ai_model_type = os.getenv("AI_MODEL_TYPE", "gemini").lower().strip()
            fallback_dims = {
                "gemini": 768,
                "openai": 1536,
                "local": 384
            }.get(ai_model_type, 768)
            return [0.0] * fallback_dims

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text using Gemini"""
        if not self.embeddings:
            logger.warning("Embeddings model not initialized. Returning zero vector as fallback.")
            # Return zero vector with 512 dimensions to match pgvector schema
            return [0.0] * 512

        try:
            embedding = self.embeddings.embed_query(text)
            # Truncate or pad to exactly 512 dimensions to match pgvector schema
            if len(embedding) > 512:
                # Truncate to 512 dimensions
                embedding = embedding[:512]
                logger.info(f"Truncated embedding from {len(embedding)} to 512 dimensions")
            elif len(embedding) < 512:
                # Pad with zeros if needed
                embedding.extend([0.0] * (512 - len(embedding)))
                logger.warning(f"Padded embedding from {len(embedding)} to 512 dimensions")

            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}. Using fallback.")
            return [0.0] * 512

    def generate_clip_text_embedding(self, text: str) -> List[float]:
      """Usa o MESMO modelo CLIP para texto (agora sim é compatível com a imagem)"""
      if not self.clip_model or not self.clip_processor:
          return [0.0] * 512

      try:
          inputs = self.clip_processor(text=text, return_tensors="pt", padding=True, truncation=True)
          with torch.no_grad():
              outputs = self.clip_model.get_text_features(**inputs)
          embedding = outputs.squeeze().tolist()
          return embedding
      except Exception as e:
          logger.error(f"Erro no CLIP text embedding: {e}")
          return [0.0] * 512

    def generate_clip_embedding(self, image_bytes: bytes) -> List[float]:
        """Generate embedding vector for image using local CLIP model"""
        if not self.clip_model or not self.clip_processor:
            logger.warning("CLIP model not initialized. Returning zero vector as fallback.")
            return [0.0] * 512

        try:
            import io

            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))

            # Process image with CLIP
            inputs = self.clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.clip_model.get_image_features(**inputs)

            embedding = outputs.squeeze().tolist()
            # CLIP embeddings are 512 dimensions, perfect match for pgvector schema
            return embedding
        except Exception as e:
            logger.error(f"Error generating CLIP image embedding: {e}. Using fallback.")
            return [0.0] * 512


    # ===================================================================
    # 2. EXTRAÇÃO DE DADOS DO DOCUMENTO (COM OCR + LANGCHAIN)
    # ===================================================================
    def _extract_data_simplified_document(self, image_path: str) -> Dict:
        """Extrai dados do documento usando OCR básico + LangChain para processamento"""
        try:
            # 1. Extrair texto da imagem usando OCR
            extracted_text = self._extract_text_from_image(image_path)

            if not extracted_text.strip():
                logger.warning("Nenhum texto extraído da imagem")
                return self._get_empty_document_data()

            # 2. Usar LangChain para processar o texto extraído e estruturar os dados
            return self._process_extracted_text_with_langchain(extracted_text)

        except Exception as e:
            logger.error(f"Erro na extração OCR: {str(e)}")
            return self._get_empty_document_data()

    def _extract_text_from_image(self, image_path: str) -> str:
        """Extrai texto da imagem usando pytesseract (OCR)"""
        try:
            import pytesseract
            import cv2
            import numpy as np

            # Abrir imagem
            image = Image.open(image_path)

            # Pré-processamento básico (converter para RGB se necessário)
            if image.mode not in ('L', 'RGB'):
                image = image.convert('RGB')

            # Melhorar contraste para OCR
            img_array = np.array(image)

            # Aplicar threshold para melhorar o contraste
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
            _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # OCR com configuração otimizada para documentos brasileiros
            custom_config = r'--oem 3 --psm 6 -l por+eng'
            text = pytesseract.image_to_string(threshold, config=custom_config, lang='por+eng')

            logger.info(f"Texto extraído da imagem: {len(text)} caracteres")
            return text

        except ImportError as e:
            logger.error(f"Biblioteca não instalada: {str(e)}. Instale com: pip install pytesseract opencv-python-headless")
            return ""
        except Exception as e:
            logger.error(f"Erro no OCR: {str(e)}")
            return ""

    def _process_extracted_text_with_langchain(self, extracted_text: str) -> Dict:
        """Processa o texto extraído usando LangChain para estruturar os dados"""
        if not self.llm:
            logger.warning("LangChain não disponível, retornando dados vazios")
            return self._get_empty_document_data()

        try:
            # Prompt para extrair dados estruturados do texto OCR
            prompt_template = """
            Você é um especialista em documentos brasileiros. Analise o texto extraído de um documento (RG, CNH, etc.) e extraia as informações em formato JSON.

            Texto extraído:
            {extracted_text}

            Retorne APENAS um JSON válido com a seguinte estrutura:
            {{
              "name": "Nome completo encontrado ou null",
              "cpf": "CPF encontrado ou null",
              "rg": "RG encontrado ou null",
              "date_of_birth": "Data de nascimento ou null",
              "mother_name": "Nome da mãe ou null",
              "father_name": "Nome do pai ou null",
              "address": {{
                "street": "Rua ou null",
                "number": "Número ou null",
                "neighborhood": "Bairro ou null",
                "city": "Cidade ou null",
                "state": "Estado (SP, RJ, etc.) ou null",
                "postal_code": "CEP ou null"
              }},
              "document_type": "RG ou CNH ou CPF ou Outro",
              "confidence": "high ou medium ou low",
              "raw_text": "texto original extraído"
            }}

            Regras:
            - Se não encontrar um campo, use null
            - Para confidence: use "high" se encontrou nome+CPF, "medium" se encontrou alguns dados, "low" se encontrou pouco
            - Limpe e formate os dados (remova caracteres especiais desnecessários)
            """

            # Usar LangChain para processar
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["extracted_text"]
            )

            chain = prompt | self.llm
            result = chain.invoke({"extracted_text": extracted_text[:2000]})  # Limitar tamanho do texto

            # Extrair o conteúdo do objeto AIMessage
            result_text = result.content if hasattr(result, 'content') else str(result)

            # Limpar possíveis markdown ou texto extra
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]

            result_text = result_text.strip()

            try:
                data = json.loads(result_text)
                data["confidence"] = data.get("confidence", "medium")
                logger.info(f"Dados extraídos com confiança: {data.get('confidence')}")
                return data

            except json.JSONDecodeError as e:
                logger.error(f"Erro ao fazer parse do JSON: {str(e)}")
                logger.error(f"Texto recebido: {result_text[:500]}")
                return self._get_empty_document_data()

        except Exception as e:
            logger.error(f"Erro no processamento com LangChain: {str(e)}")
            return self._get_empty_document_data()

    def _get_empty_document_data(self) -> Dict:
        """Retorna estrutura vazia para documento não processado"""
        return {
            "name": None,
            "cpf": None,
            "rg": None,
            "date_of_birth": None,
            "mother_name": None,
            "father_name": None,
            "address": {
                "street": None,
                "number": None,
                "neighborhood": None,
                "city": None,
                "state": None,
                "postal_code": None
            },
            "document_type": "desconhecido",
            "confidence": "low",
            "note": "Falha na extração OCR - usar dados manuais",
            "raw_text": ""
        }

    def analyze_query_threshold(self, query: str) -> float:
        """
        Usa IA para analisar uma query de busca e determinar o threshold ideal de similaridade.
        Retorna um valor float entre 0.0 e 1.0.
        """
        if not self.llm:
            logger.warning("LLM não disponível, usando threshold padrão")
            return 0.35

        try:
            analysis_prompt = f"""
            Eu sou um Agente especializado em procurar um ótimo threshold e baseado na pesquisa a seguir gostaria de um possível numero a usar que buscaria os melhores resultados:

            Query: "{query}"

            Analise esta query de busca e determine o threshold ideal de similaridade (0.0 a 1.0) para busca vetorial em um sistema de busca de clientes.

            Considere:
            - Queries específicas (nomes próprios, emails, CPFs, documentos): threshold mais alto (0.7-0.9)
            - Queries descritivas gerais ("cliente de SP", "pessoa jovem", endereços): threshold médio (0.3-0.6)
            - Queries muito genéricas ("cliente", "pessoa", uma palavra só): threshold baixo (0.1-0.3)

            Retorne apenas um número decimal entre 0.0 e 1.0 representando o threshold recomendado.
            Não inclua explicações, apenas o número.
            """

            # Usar LangChain para fazer a análise
            prompt = PromptTemplate(
                template=analysis_prompt,
                input_variables=[]
            )

            chain = prompt | self.llm
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
                logger.info(f"IA determinou threshold {threshold} para query: '{query}'")
                return threshold

        except Exception as e:
            logger.warning(f"Erro ao consultar IA sobre threshold: {e}")

        # Fallback para heurísticas se IA falhar
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

    def process_text_with_custom_prompt(self, text_content: str, custom_prompt: str) -> Dict:
        """
        Processa texto usando um prompt customizado com LangChain
        Retorna dados extraídos em formato de dicionário
        """
        try:
            prompt = PromptTemplate(
                template=custom_prompt,
                input_variables=["text_content"]
            )

            chain = prompt | self.llm
            result = chain.invoke({"text_content": text_content[:4000]})  # Limitar tamanho do texto

            # Extrair o conteúdo do objeto AIMessage
            result_text = result.content if hasattr(result, 'content') else str(result)

            # Limpar possíveis markdown ou texto extra
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]

            result_text = result_text.strip()

            try:
                # Tentar fazer parse como JSON
                data = json.loads(result_text)
                return data

            except json.JSONDecodeError:
                # Se não for JSON, retornar como texto simples
                logger.warning(f"Resposta não é JSON válido: {result_text[:200]}")
                return {
                    "raw_response": result_text,
                    "error": "Resposta não está em formato JSON esperado"
                }

        except Exception as e:
            logger.error(f"Erro no processamento customizado: {str(e)}")
            return self._get_empty_document_data()

    # ===================================================================
    # 5. BUSCA NO FILE SEARCH STORE (stub - não disponível na versão atual)
    # ===================================================================
    def search_files_in_store(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Busca arquivos no File Search Store
        Como não está disponível na versão atual do SDK, retorna lista vazia
        """
        logger.warning("File Search Store não disponível na versão atual do SDK")
        return []

    # ===================================================================
    # 6. PROCESSAR IMAGEM (para embeddings e descrições)
    # ===================================================================
    def process_image(self, image_path: str, user_description: str = None) -> tuple:
        """
        Processa imagem para gerar embedding e descrição usando Gemini
        Retorna (embedding, description)
        """
        try:
            # 1. Gerar descrição rica (Gemini)
            description = self._generate_description_gemini(image_path, user_description)

            # 2. Gerar Embedding do texto (Gemini)
            full_text = f"{user_description or ''} {description}"
            embedding = self.generate_embedding(full_text)

            logger.info(f"Imagem processada com Gemini: {os.path.basename(image_path)}")
            return embedding, description

        except Exception as e:
            logger.error(f"Erro ao processar imagem {image_path}: {str(e)}")
            # Retornar valores vazios em caso de erro para não quebrar o worker
            return [], "Erro no processamento"

    def _generate_image_description(self, image_path: str, user_description: str = None) -> str:
        """
        Gera descrição da imagem usando LangChain (Gemini ou modelo local)
        """
        if not self.llm:
            return f"Descrição não disponível - modelo LangChain não inicializado{f' - {user_description}' if user_description else ''}"

        try:
            model_type = os.getenv("AI_MODEL_TYPE", "local").lower()

            if model_type == "gemini":
                return self._generate_description_gemini(image_path, user_description)
            else:
                return self._generate_description_local(image_path, user_description)

        except Exception as e:
            logger.error(f"Erro ao gerar descrição com LangChain: {str(e)}")
            return f"Erro na geração da descrição{f' - {user_description}' if user_description else ''}"

    def _generate_description_gemini(self, image_path: str, user_description: str = None) -> str:
        """Gera descrição usando Gemini (multimodal)"""
        try:
            # Preparing the image
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type:
                mime_type = "image/jpeg"

            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")

            # 2. Prepare the prompt
            prompt_text = """
            Você é um especialista em análise de imagens. Descreva esta imagem em detalhes.
            Inclua:
            - Objetos principais e secundários
            - Cores predominantes e estilo
            - Texto visível (se houver)
            - Contexto e ambiente

            Responda em PORTUGUÊS do Brasil.
            """

            if user_description:
                prompt_text += f"\nContexto adicional fornecido pelo usuário: {user_description}"

            # 3. Create message with image
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_data}"}
                    }
                ]
            )

            # 4. Invoke model
            response = self.llm.invoke([message])
            return response.content if hasattr(response, 'content') else str(response)

        except Exception as e:
            logger.error(f"Erro com Gemini Multimodal: {str(e)}")
            # Fallback to local if Gemini fails (e.g. API error)
            return self._generate_description_local(image_path, user_description)

    def _generate_description_local(self, image_path: str, user_description: str = None) -> str:
        """Gera descrição usando modelo local (text-only)"""
        # Prompt customizado para descrição de imagens
        prompt_template = """
        Você é um especialista em análise de imagens. Baseado na descrição fornecida, crie uma descrição detalhada e rica da imagem.

        Descrição da imagem: {image_info}
        {user_context}

        Gere uma descrição completa em português que inclua:
        - Objetos e elementos principais
        - Cores e composição
        - Atmosfera e estilo
        - Detalhes específicos
        - Contexto e interpretação

        Descrição:
        """

        # Como não temos modelo multimodal, usar uma descrição básica da imagem
        image_info = f"Arquivo: {os.path.basename(image_path)} (tipo: {self._get_image_type(image_path)})"

        if user_description:
            user_context = f"Contexto adicional do usuário: {user_description}"
        else:
            user_context = ""

        # Usar LangChain para gerar a descrição com sintaxe moderna
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["image_info", "user_context"]
        )

        chain = prompt | self.llm
        result = chain.invoke({"image_info": image_info, "user_context": user_context})

        # Limpar e formatar a resposta
        description = result.content if hasattr(result, 'content') else str(result)
        if description.startswith("Descrição:"):
            description = description[11:].strip()

        return description

    def _get_image_type(self, image_path: str) -> str:
        """Retorna o tipo da imagem baseado na extensão"""
        ext = os.path.splitext(image_path)[1].lower()
        return {
            '.jpg': 'JPEG',
            '.jpeg': 'JPEG',
            '.png': 'PNG',
            '.gif': 'GIF',
            '.bmp': 'BMP'
        }.get(ext, 'desconhecido')
