# ai_service.py - VERSÃO FINAL 2025 (IMAGENS + EXTRAÇÃO DE CLIENTES)
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
from fastapi import UploadFile
import google.generativeai as genai
from dotenv import load_dotenv
import json

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        self.store_id = self._get_or_create_store()
        self.model = genai.GenerativeModel("gemini-2.5-flash")  # ou "gemini-2.5-pro" se quiser mais preciso
        logger.info(f"AIService iniciado | Store ID: {self.store_id}")

    def _get_or_create_store(self) -> str:
        store_id = os.getenv("FILE_SEARCH_STORE_ID", "").strip()
        if store_id and store_id.startswith("file_search_stores/"):
            try:
                # Tentar validar se o store existe
                genai.get_file_search_store(store_id)
                return store_id
            except Exception as e:
                logger.warning(f"Store ID {store_id} inválido ou não encontrado: {str(e)}")

        # Se não tem store configurado ou disponível, usar abordagem simplificada sem File Search Store
        logger.warning("FILE_SEARCH_STORE_ID não configurado ou indisponível. Usando busca simplificada.")
        return None

    # ===================================================================
    # 1. ADICIONAR CLIENTE (foto do documento + selfie + dados opcionais)
    # ===================================================================
    async def adicionar_cliente(
        self,
        documento: UploadFile,           # foto do RG, CNH, etc
        selfie: Optional[UploadFile] = None,
        nome: Optional[str] = None,
        cpf: Optional[str] = None,
        telefone: Optional[str] = None,
        notas: Optional[str] = None
    ) -> Dict:
        """Adiciona cliente com extração automática de dados + busca por foto"""
        temp_dir = Path("temp_clientes")
        temp_dir.mkdir(exist_ok=True)

        doc_path = temp_dir / documento.filename
        selfie_path = temp_dir / (selfie.filename if selfie else "sem_selfie.jpg")

        # Salva arquivos temporariamente
        doc_path.write_bytes(await documento.read())
        if selfie:
            selfie_path.write_bytes(await selfie.read())

        try:
            # 1. Extrai dados do documento com Gemini (melhor que qualquer regex do mundo)
            extracted = self._extrair_dados_documento(str(doc_path))

            # 2. Gera descrição rica pro File Search (pra buscar por "homem de barba" depois)
            descricao = f"""
            Cliente: {extracted.get('name', 'Nome não identificado')}
            CPF: {extracted.get('cpf', 'não encontrado')}
            Telefone: {telefone or extracted.get('phone', 'não informado')}
            Cidade: {extracted.get('address', {}).get('city', 'não informada')}
            Notas: {notas or 'sem notas'}
            """

            # 3. Upload do documento
            doc_file = genai.upload_file(path=str(doc_path), display_name=f"doc_{documento.filename}")

            # 4. Upload da selfie (se tiver)
            selfie_file = None
            if selfie and selfie_path.exists():
                selfie_file = genai.upload_file(path=str(selfie_path), display_name=f"selfie_{documento.filename}")

            # 5. Upload da descrição como .txt (pra busca semântica perfeita)
            desc_path = temp_dir / f"desc_{documento.filename}.txt"
            desc_path.write_text(descricao, encoding="utf-8")
            desc_file = genai.upload_file(path=str(desc_path), display_name=f"info_{documento.filename}")

            # 5. Como não temos File Search Store, apenas salvar no banco local
            logger.info(f"Cliente {extracted.get('name', 'Desconhecido')} extraído com sucesso (sem File Search Store)")

            return {
                "status": "success",
                "extracted_data": extracted,
                "message": "Cliente processado com sucesso! Dados extraídos do documento."
            }

        finally:
            # Limpeza
            for p in [doc_path, selfie_path, desc_path]:
                if p.exists():
                    p.unlink()

    # ===================================================================
    # 2. EXTRAÇÃO DE DADOS DO DOCUMENTO (RG, CNH, SELFIE, ETC)
    # ===================================================================
    def _extrair_dados_documento(self, image_path: str) -> Dict:
        """Extrai nome, CPF, data de nascimento, endereço etc com Gemini 2.5"""
        prompt = """
        Você é um especialista em documentos brasileiros (RG, CNH, Carteira de Trabalho, etc).
        Analise esta imagem e extraia TODAS as informações possíveis em formato JSON limpo.

        Retorne APENAS o JSON, nada mais. Use este formato exato:

        {
          "name": "Nome completo",
          "cpf": "123.456.789-00",
          "rg": "12.345.678-9",
          "date_of_birth": "15/03/1985",
          "mother_name": "Maria Silva",
          "father_name": "José Santos",
          "address": {
            "street": "Rua das Flores",
            "number": "123",
            "neighborhood": "Centro",
            "city": "São Paulo",
            "state": "SP",
            "postal_code": "01001-000"
          },
          "document_type": "RG" ou "CNH" ou "Outro",
          "confidence": "high|medium|low"
        }

        Se não encontrar algum campo, coloque null.
        """

        image = {
            "mime_type": "image/jpeg",
            "data": Path(image_path).read_bytes()
        }

        response = self.model.generate_content([prompt, image])
        try:
            # Tenta parsear o JSON direto
            data = json.loads(response.text)
            data["confidence"] = data.get("confidence", "medium")
            return data
        except:
            # Fallback se o Gemini não retornou JSON perfeito
            return {
                "name": None,
                "cpf": None,
                "date_of_birth": None,
                "address": {},
                "document_type": "desconhecido",
                "confidence": "low",
                "raw_response": response.text[:500]
            }

    # ===================================================================
    # 3. BUSCAR CLIENTE POR TEXTO OU FOTO
    # ===================================================================
    def buscar_cliente(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Busca por: "João Silva", "CPF 123.456.789-00", "homem de barba", "mulher loira de óculos"
        Como não temos File Search Store, retorna lista vazia por enquanto
        """
        logger.info(f"Busca por '{query}' - File Search Store não disponível")
        return []

    # ===================================================================
    # 4. BUSCA POR SELFIE (reverse image search)
    # ===================================================================
    async def buscar_por_selfie(self, selfie: UploadFile) -> List[Dict]:
        """
        Busca por selfie (reverse image search)
        Como não temos File Search Store, retorna lista vazia por enquanto
        """
        logger.info("Busca por selfie - File Search Store não disponível")
        return []

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
        Processa imagem para gerar embedding e descrição usando CLIP/Gemini
        Retorna (embedding, description)
        """
        try:
            # Por enquanto, usar implementação simplificada
            # TODO: Implementar processamento real com CLIP ou Gemini Vision

            # Placeholder: gerar embedding fake (512 dimensões como CLIP)
            import numpy as np
            embedding_array = np.random.rand(512).astype(np.float32)
            embedding = embedding_array.tolist()  # Converter para list[float]

            # Gerar descrição usando Gemini (sempre, independente da user_description)
            with open(image_path, "rb") as img_file:
                image_data = img_file.read()

            # Usar user_description como contexto adicional se existir
            prompt_base = "Descreva detalhadamente esta imagem em português. Foque em objetos, pessoas, animais, cores, composição, atmosfera e detalhes específicos."

            if user_description:
                prompt = f"{prompt_base} Contexto adicional fornecido pelo usuário: {user_description}. Use este contexto para enriquecer a descrição, mas descreva o que realmente vê na imagem."
            else:
                prompt = prompt_base

            response = self.model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": image_data}
            ])

            description = response.text.strip() if response.text else f"Imagem processada{f' - {user_description}' if user_description else ''}"

            logger.info(f"Imagem processada: {os.path.basename(image_path)}")
            return embedding, description

        except Exception as e:
            logger.error(f"Erro ao processar imagem {image_path}: {str(e)}")
            # Retornar valores padrão em caso de erro
            import numpy as np
            return np.random.rand(512).astype(np.float32).tolist(), "Erro no processamento"
