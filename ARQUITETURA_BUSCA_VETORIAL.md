# Arquitetura de Busca Vetorial Híbrida

## Visão Geral

Este sistema implementa uma **arquitetura híbrida** para busca vetorial, combinando **PostgreSQL com pgvector** e **ChromaDB** para fornecer diferentes níveis de busca semântica otimizados para casos de uso específicos.

## Componentes da Arquitetura

### 1. PostgreSQL com pgvector (VectorService)

**Responsabilidades:**

- ✅ Armazenamento primário de dados relacionais
- ✅ Busca vetorial básica e rápida
- ✅ Integração nativa com queries SQL
- ✅ Backup transacional (ACID) dos dados

**Implementação:**

```python
# Armazenamento direto na tabela Photo
photo.embedding = embedding  # Vetor de 512 dimensões (CLIP)
photo.description = description  # Descrição gerada por IA
photo.processed = True
db.commit()
```

**Casos de uso:**

- Busca simples por similaridade
- Integração com queries relacionais
- Cenários onde velocidade é prioridade

### 2. ChromaDB (VisualSearchService)

**Responsabilidades:**

- 🎯 Busca semântica avançada com re-ranking
- 🎯 Busca multimodal (imagem + texto)
- 🎯 Filtragem inteligente de falsos positivos
- 🎯 Metadados ricos e contexto aprimorado

**Implementação:**

```python
# Índice separado otimizado para busca avançada
self.collection.add(
    ids=[doc_id],
    embeddings=[combined_embedding],  # Mesmo vetor, mas otimizado
    documents=[rich_caption],  # Caption rica gerada por LLM
    metadatas=[comprehensive_metadata]  # Metadados extras
)
```

**Casos de uso:**

- Busca precisa com entendimento semântico
- Eliminação de falsos positivos
- Experiências de busca mais "inteligentes"

## Fluxo de Processamento

```
📸 Foto populada/uploadada via API
   ↓
🤖 Worker processa (photo_processor.py):
   ├── PostgreSQL: Salva embedding + description básica
   └── ChromaDB: Cria índice avançado com caption rica
   ↓
🔍 Busca pode usar diferentes estratégias:
   ├── VectorService: Busca rápida no PostgreSQL
   └── VisualSearchService: Busca inteligente no ChromaDB
```

## Comparação Técnica

| Aspecto | PostgreSQL (pgvector) | ChromaDB |
|---------|----------------------|----------|
| **Velocidade** | ⚡ Muito rápida | 🐌 Mais lenta (re-ranking) |
| **Precisão** | 📊 Boa | 🎯 Excelente (com LLM) |
| **Integração** | 🔗 Nativa com SQL | 📦 Serviço separado |
| **Armazenamento** | 💾 Junto aos dados | 💾 Índice especializado |
| **Backup** | ✅ Automático | ⚠️ Manual necessário |
| **Complexidade** | 🔧 Simples | 🧠 Complexa |

## Quando Usar Cada Um

### Use VectorService (PostgreSQL)

```python
# Para casos simples e integrados
photo_service.search_similar_photos(query_text="gato")
# → Busca direta na tabela Photo usando pgvector
```

### Use VisualSearchService (ChromaDB)

```python
# Para busca avançada e inteligente
GET /api/v1/photos/search/text?q=gato%20na%20praia
# → Busca semântica com re-ranking por LLM
```

## Vantagens da Arquitetura Híbrida

### PostgreSQL + pgvector

- **Confiabilidade**: Dados críticos ficam no banco relacional
- **Performance**: Busca rápida para casos comuns
- **Simplicidade**: Integração natural com o resto da aplicação
- **Backup**: Automaticamente incluído nos backups do banco

### ChromaDB

- **Inteligência**: Usa LLMs para melhorar resultados
- **Precisão**: Elimina falsos positivos através de re-ranking
- **Flexibilidade**: Permite metadados ricos e busca multimodal
- **Especialização**: Otimizado especificamente para busca vetorial

## Implementação Prática

### Processamento de Fotos

```python
# 1. Worker processa a foto
ai_service = AIService()
embedding, description = ai_service.process_image(file_path, user_description)

# 2. Salva no PostgreSQL
photo.embedding = embedding
photo.description = description
photo.processed = True
db.commit()

# 3. Indexa no ChromaDB
visual_search = VisualSearchService()
visual_search.add_image(file_path, photo.id, user_description)
```

### Busca por Texto

```python
# Busca simples (PostgreSQL)
vector_service = VectorService(db)
results = vector_service.search_similar_photos("gato", limit=10)

# Busca avançada (ChromaDB)
visual_search = VisualSearchService()
results = visual_search.search_by_text("gato brincando", top_k=10)
```

## Considerações de Produção

### Monitoramento

- Monitorar performance de ambas as buscas
- Alertas se ChromaDB ficar desincronizado
- Backup regular do diretório `data/chroma_db/`

### Manutenção

- Sincronização entre PostgreSQL e ChromaDB
- Reindexação periódica se necessário
- Atualização de embeddings quando modelos mudam

### Escalabilidade

- PostgreSQL escala horizontalmente com replicas
- ChromaDB pode ser distribuído, mas é mais complexo
- Considerar cache para resultados frequentes

## Conclusão

Esta arquitetura híbrida permite **o melhor dos dois mundos**:

- **Velocidade e confiabilidade** do PostgreSQL para casos comuns
- **Inteligência e precisão** do ChromaDB para buscas avançadas

A escolha entre VectorService e VisualSearchService depende do caso de uso específico e dos requisitos de precisão vs. velocidade.
