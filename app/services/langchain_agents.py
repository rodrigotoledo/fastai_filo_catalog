from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ImageProcessingAgent:
    """Agent responsável por processar imagens e decidir estratégias"""

    def __init__(self, llm, visual_search_service, ai_service, photo_service=None):
        self.llm = llm
        self.visual_search = visual_search_service
        self.ai_service = ai_service
        self.photo_service = photo_service
        self.agent_executor = self._create_agent()

    def _create_agent(self):
        """Cria o agent com ferramentas específicas para processamento de imagens"""

        # Ferramentas disponíveis para o agent
        tools = [
            ProcessImageTool(visual_search_service=self.visual_search, ai_service=self.ai_service),
            GenerateRichCaptionTool(ai_service=self.ai_service),
            SearchSimilarImagesTool(visual_search_service=self.visual_search),
            GetCollectionStatsTool(visual_search_service=self.visual_search),
            SmartSearchTool(visual_search_service=self.visual_search, photo_service=self.photo_service)
        ]

        # Prompt do sistema para o agent
        system_prompt = """Você é um especialista em processamento de imagens e busca visual.

Sua tarefa é processar imagens de forma inteligente, decidindo as melhores estratégias:

1. **Processamento Inicial**:
   - Gerar embeddings para busca posterior
   - Criar descrições ricas da imagem
   - Indexar no sistema de busca visual

2. **Busca e Similaridade**:
   - Buscar imagens similares por texto
   - Buscar imagens similares por imagem
   - Expandir consultas para melhor matching

3. **Análise e Decisão**:
   - Avaliar qualidade da imagem
   - Decidir se precisa de processamento adicional
   - Otimizar para diferentes tipos de conteúdo

Use as ferramentas disponíveis de forma estratégica. Sempre explique seu raciocínio."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_functions_agent(self.llm, tools, prompt)
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )

    def process_image(self, image_path: str, user_description: str = None) -> Dict[str, Any]:
        """Processa uma imagem usando o agent"""

        task_description = f"""
        Processar a imagem localizada em: {image_path}

        Contexto adicional do usuário: {user_description or 'Nenhum contexto fornecido'}

        Tarefas a executar:
        1. Processar a imagem para gerar embedding e descrição
        2. Indexar no sistema de busca visual
        3. Verificar se o processamento foi bem-sucedido
        4. Retornar estatísticas da coleção após indexação

        Execute essas tarefas em sequência lógica.
        """

        try:
            result = self.agent_executor.invoke({"input": task_description})
            return {
                "success": True,
                "agent_response": result.get("output", ""),
                "intermediate_steps": result.get("intermediate_steps", [])
            }
        except Exception as e:
            logger.error(f"Erro no agent de processamento de imagem: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "fallback_result": self._fallback_processing(image_path, user_description)
            }

    def _fallback_processing(self, image_path: str, user_description: str):
        """Processamento fallback se o agent falhar"""
        try:
            # Processamento direto sem agent
            embedding, description = self.ai_service.process_image(image_path, user_description)
            photo_id = self.visual_search.add_image(image_path, 0, user_description)  # photo_id será definido depois
            stats = self.visual_search.get_collection_stats()

            return {
                "embedding_generated": True,
                "description": description,
                "indexed": True,
                "collection_stats": stats
            }
        except Exception as e:
            return {"error": f"Fallback também falhou: {str(e)}"}


class DocumentProcessingAgent:
    """Agent responsável por extrair dados de documentos"""

    def __init__(self, llm, document_parser_service, ai_service):
        self.llm = llm
        self.document_parser = document_parser_service
        self.ai_service = ai_service
        self.agent_executor = self._create_agent()

    def _create_agent(self):
        """Cria o agent com ferramentas para processamento de documentos"""

        tools = [
            ExtractDocumentDataTool(document_parser_service=self.document_parser),
            ValidateExtractedDataTool(document_parser_service=self.document_parser),
            OCRTextExtractionTool(ai_service=self.ai_service),
            StructureDataTool(ai_service=self.ai_service)
        ]

        system_prompt = """Você é um especialista em processamento de documentos brasileiros.

Sua tarefa é extrair e estruturar dados de documentos de forma inteligente:

1. **Análise Inicial**:
   - Identificar tipo de documento (RG, CNH, CPF, etc.)
   - Escolher melhor estratégia de extração

2. **Extração de Texto**:
   - Usar OCR quando necessário
   - Processar diferentes formatos (PDF, imagem, etc.)

3. **Estruturação de Dados**:
   - Extrair informações específicas (nome, CPF, endereço, etc.)
   - Validar dados extraídos
   - Corrigir inconsistências

4. **Validação e Qualidade**:
   - Verificar integridade dos dados
   - Avaliar confiança da extração
   - Sugerir correções quando necessário

Seja meticuloso e pergunte se precisar de esclarecimentos."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_functions_agent(self.llm, tools, prompt)
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )

    def process_document(self, file_path: str, filename: str, extraction_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Processa um documento usando o agent"""

        task_description = f"""
        Processar o documento: {filename}
        Localizado em: {file_path}

        Prompt de extração personalizado: {extraction_prompt or 'Usar extração padrão'}

        Tarefas:
        1. Identificar tipo de documento
        2. Extrair texto usando método apropriado
        3. Estruturar dados em formato JSON
        4. Validar dados extraídos
        5. Retornar resultado com nível de confiança

        Execute de forma sistemática e relate qualquer problema encontrado.
        """

        try:
            result = self.agent_executor.invoke({"input": task_description})
            return {
                "success": True,
                "agent_response": result.get("output", ""),
                "structured_data": self._extract_structured_data_from_response(result)
            }
        except Exception as e:
            logger.error(f"Erro no agent de processamento de documento: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "fallback_result": self._fallback_processing(file_path, filename, extraction_prompt)
            }

    def _extract_structured_data_from_response(self, agent_result) -> Dict:
        """Extrai dados estruturados da resposta do agent"""
        # Implementar lógica para parsear resposta do agent
        return {}

    def _fallback_processing(self, file_path: str, filename: str, extraction_prompt: str):
        """Processamento fallback direto"""
        try:
            result = self.document_parser.parse_document(file_path, filename, extraction_prompt)
            validation_errors = self.document_parser.validate_extracted_data(result)

            return {
                "extracted_data": result,
                "validation_errors": validation_errors,
                "confidence": result.get("confidence", "low")
            }
        except Exception as e:
            return {"error": f"Fallback falhou: {str(e)}"}


# Ferramentas para os Agents (implementação básica)
class ProcessImageTool(BaseTool):
    name: str = "process_image"
    description: str = "Processa uma imagem para gerar embedding e descrição"

    visual_search_service: Any = None
    ai_service: Any = None

    def __init__(self, visual_search_service, ai_service):
        super().__init__()
        self.visual_search_service = visual_search_service
        self.ai_service = ai_service

    def _run(self, image_path: str) -> str:
        embedding, description = self.ai_service.process_image(image_path)
        return f"Imagem processada. Embedding: {len(embedding)} dims, Descrição: {description[:100]}..."

class GenerateRichCaptionTool(BaseTool):
    name: str = "generate_rich_caption"
    description: str = "Gera uma descrição rica e detalhada da imagem"

    ai_service: Any = None

    def __init__(self, ai_service):
        super().__init__()
        self.ai_service = ai_service

    def _run(self, image_path: str, user_context: str = None) -> str:
        caption = self.ai_service._generate_rich_caption(image_path, user_context)
        return f"Legenda gerada: {caption}"

class SearchSimilarImagesTool(BaseTool):
    name: str = "search_similar_images"
    description: str = "Busca imagens similares por texto ou imagem"

    visual_search_service: Any = None

    def __init__(self, visual_search_service):
        super().__init__()
        self.visual_search_service = visual_search_service

    def _run(self, query: str, search_type: str = "text", top_k: int = 5) -> str:
        if search_type == "text":
            results = self.visual_search_service.search_by_text(query, top_k)
        else:
            results = self.visual_search_service.search_by_image(query, top_k)

        return f"Encontrados {len(results)} resultados similares"

class GetCollectionStatsTool(BaseTool):
    name: str = "get_collection_stats"
    description: str = "Obtém estatísticas da coleção de imagens"

    visual_search_service: Any = None

    def __init__(self, visual_search_service):
        super().__init__()
        self.visual_search_service = visual_search_service

    def _run(self) -> str:
        stats = self.visual_search_service.get_collection_stats()
        return f"Collection stats: {stats}"

class SmartSearchTool(BaseTool):
    name: str = "smart_search"
    description: str = "Realiza busca inteligente de imagens usando múltiplas estratégias"

    visual_search_service: Any = None
    photo_service: Any = None

    def __init__(self, visual_search_service, photo_service):
        super().__init__()
        self.visual_search_service = visual_search_service
        self.photo_service = photo_service

    def _run(self, query: str, strategy: str = "semantic", top_k: int = 50) -> str:
        """
        Executa busca inteligente baseada na estratégia escolhida
        """
        try:
            if strategy == "semantic":
                # Busca semântica padrão
                results = self.visual_search_service.search_by_text(query, top_k=top_k)
            elif strategy == "expanded":
                # Expandir query com sinônimos (simplificado)
                expanded_queries = self._expand_query(query)
                all_results = []
                for q in expanded_queries:
                    results = self.visual_search_service.search_by_text(q, top_k=top_k//len(expanded_queries))
                    all_results.extend(results)

                # Remover duplicatas e ordenar por similaridade
                seen_ids = set()
                unique_results = []
                for result in sorted(all_results, key=lambda x: x['similarity'], reverse=True):
                    if result['photo_id'] not in seen_ids:
                        unique_results.append(result)
                        seen_ids.add(result['photo_id'])
                        if len(unique_results) >= top_k:
                            break
                results = unique_results
            else:
                results = self.visual_search_service.search_by_text(query, top_k=top_k)

            # Formatar resultados para o agente
            formatted_results = []
            for result in results[:10]:  # Limitar para resposta do agente
                photo = self.photo_service.get_photo(result['photo_id'])
                if photo:
                    formatted_results.append({
                        'id': photo.id,
                        'filename': photo.original_filename,
                        'description': photo.description[:200] + '...' if photo.description and len(photo.description) > 200 else photo.description,
                        'similarity': result['similarity']
                    })

            return f"Encontrados {len(results)} resultados. Top 10: {formatted_results}"

        except Exception as e:
            return f"Erro na busca: {str(e)}"

    def _expand_query(self, query: str) -> List[str]:
        """
        Expande a query com sinônimos e variações (simplificado)
        """
        expansions = [query]

        # Mapeamentos simples de sinônimos
        synonyms = {
            'gato': ['cat', 'felino', 'gatinho'],
            'cachorro': ['dog', 'cão', 'cachorrinho'],
            'carro': ['veículo', 'automóvel', 'car'],
            'comida': ['alimento', 'refeição', 'prato']
        }

        for word in query.lower().split():
            if word in synonyms:
                for synonym in synonyms[word]:
                    expansions.append(query.lower().replace(word, synonym))

        return list(set(expansions))  # Remover duplicatas

class ExtractDocumentDataTool(BaseTool):
    name: str = "extract_document_data"
    description: str = "Extrai dados estruturados de um documento"

    document_parser_service: Any = None

    def __init__(self, document_parser_service):
        super().__init__()
        self.document_parser_service = document_parser_service

    def _run(self, file_path: str, filename: str) -> str:
        result = self.document_parser_service.parse_document(file_path, filename)
        return f"Dados extraídos: {result}"

class ValidateExtractedDataTool(BaseTool):
    name: str = "validate_extracted_data"
    description: str = "Valida dados extraídos de documento"

    document_parser_service: Any = None

    def __init__(self, document_parser_service):
        super().__init__()
        self.document_parser_service = document_parser_service

    def _run(self, data: Dict) -> str:
        errors = self.document_parser_service.validate_extracted_data(data)
        if errors:
            return f"Erros de validação encontrados: {errors}"
        return "Dados válidos"

class OCRTextExtractionTool(BaseTool):
    name: str = "ocr_text_extraction"
    description: str = "Extrai texto de imagem usando OCR"

    ai_service: Any = None

    def __init__(self, ai_service):
        super().__init__()
        self.ai_service = ai_service

    def _run(self, image_path: str) -> str:
        text = self.ai_service._extract_text_from_image(image_path)
        return f"Texto extraído: {text[:200]}..."

class StructureDataTool(BaseTool):
    name: str = "structure_data"
    description: str = "Estrutura dados extraídos usando LLM"

    ai_service: Any = None

    def __init__(self, ai_service):
        super().__init__()
        self.ai_service = ai_service

    def _run(self, raw_text: str) -> str:
        structured = self.ai_service._process_extracted_text_with_langchain(raw_text)
        return f"Dados estruturados: {structured}"
