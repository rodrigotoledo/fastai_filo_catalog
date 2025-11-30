# Photo Finder Backend – FastAPI AI API

API backend robusto, construído com [FastAPI](https://fastapi.tiangolo.com/), integrando recursos de IA (Machine Learning/NLP) e banco de dados PostgreSQL. Ideal para aplicações que exigem busca, análise e manipulação inteligente de fotos/imagens.

---

## Sumário

- [Photo Finder Backend – FastAPI AI API](#photo-finder-backend--fastapi-ai-api)
  - [Sumário](#sumário)
  - [Funcionalidades](#funcionalidades)
  - [Requisitos](#requisitos)
  - [Instalação e Setup](#instalação-e-setup)
  - [Configuração de Ambiente](#configuração-de-ambiente)
  - [Execução](#execução)
  - [Testes](#testes)
  - [Estrutura do Projeto (sugerida)](#estrutura-do-projeto-sugerida)
  - [Principais Dependências](#principais-dependências)
  - [Deploy](#deploy)
  - [Referências](#referências)

---

## Funcionalidades

- Endpoints RESTful com FastAPI
- Integração de IA: modelos de ML/NLP, embeddings, busca vetorial
- Banco de dados PostgreSQL com suporte a vetores (pgvector)
- Fila de tarefas assíncronas com Redis/RQ
- Documentação automática via Swagger/OpenAPI

## Requisitos

- Python 3.8+
- PostgreSQL 13+
- Redis (opcional, para filas)
- pip

## Instalação e Setup

1. **Clone o repositório**
   ```bash
   git clone https://github.com/yourusername/photo-finder.git
   cd photo-finder/backend
   ```

2. **Crie e ative o ambiente virtual**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   ```

3. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Opcional) Gere um novo requirements.txt limpo**
   ```bash
   pip install pipreqs
   pipreqs . --force
   ```

## Configuração de Ambiente

Crie um arquivo `.env` com as variáveis de ambiente necessárias:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/yourdatabase
# Adicione outras variáveis conforme necessário
```

## Execução

1. **(Opcional) Migrações de banco de dados**
   ```bash
   alembic upgrade head
   ```

2. **Inicie o servidor FastAPI**
   ```bash
   uvicorn app.main:app --reload
   ```

3. **Acesse a documentação interativa**
   - [http://localhost:8000/docs](http://localhost:8000/docs)

## Testes

1. **Execute os testes (exemplo com pytest):**
   ```bash
   pytest
   ```
   > Adapte conforme a ferramenta de testes utilizada.

## Estrutura do Projeto (sugerida)

```text
backend/
├── app/
│   ├── main.py         # Ponto de entrada FastAPI
│   ├── api/           # Rotas e endpoints
│   ├── models/        # Modelos Pydantic/SQLModel
│   ├── services/      # Lógica de negócio e IA
│   ├── db/            # Configuração e scripts do banco
│   └── ...
├── requirements.txt
├── README.md
├── .env
└── ...
```

## Principais Dependências

- **FastAPI**: Framework web moderno e rápido
- **SQLModel**: ORM para SQL/NoSQL
- **pgvector**: Suporte a vetores no PostgreSQL
- **sentence-transformers, transformers, torch**: Modelos de IA e embeddings
- **Redis, rq**: Fila de tarefas assíncronas
- **python-dotenv**: Gerenciamento de variáveis de ambiente

Veja todas as dependências em [`requirements.txt`](requirements.txt).

## Deploy

Sugestão de deploy (exemplo com Docker):

```dockerfile
# Dockerfile exemplo
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --upgrade pip && pip install -r requirements.txt
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Referências

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [SQLModel](https://sqlmodel.tiangolo.com/)
- [pgvector](https://github.com/pgvector/pgvector)
- [Sentence Transformers](https://www.sbert.net/)
- [Uvicorn](https://www.uvicorn.org/)

---
Sinta-se à vontade para contribuir ou sugerir melhorias!
