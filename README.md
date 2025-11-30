# Photo Finder Backend – FastAPI AI API

API backend robusto, construído com [FastAPI](https://fastapi.tiangolo.com/), integrando recursos de IA (Machine Learning/NLP) e banco de dados PostgreSQL. Ideal para aplicações que exigem busca, análise e manipulação inteligente de fotos/imagens.

---

## Sumário

- [Funcionalidades](#funcionalidades)
- [Requisitos](#requisitos)
- [Instalação e Setup](#instalação-e-setup)
- [Execução](#execução)
- [Testes](#testes)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Principais Dependências](#principais-dependências)
- [Deploy](#deploy)
- [Referências](#referências)

---

## Funcionalidades

- Endpoints RESTful com FastAPI
- Integração de IA: modelos de ML/NLP, embeddings, busca vetorial com pgvector
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

5. **Configure o banco de dados com Docker**

   ```bash
   docker compose up -d
   ```

   > Isso iniciará o PostgreSQL e Redis em containers Docker.

6. **Configure as variáveis de ambiente**

   Copie o `.env.example` para `.env` e ajuste conforme necessário:

   ```bash
   cp .env.example .env
   ```

## Configuração de Ambiente

Copie o `.env.example` para `.env` e ajuste conforme necessário:

```bash
cp .env.example .env
```

O arquivo `.env` contém as variáveis necessárias, como `DATABASE_URL` e `REDIS_URL`.

## Execução

1. **Migrações de banco de dados** (após criar modelos)
   ```bash
   # Criar uma nova migração
   alembic revision --autogenerate -m "Descrição da migração"

   # Aplicar migrações
   alembic upgrade head
   ```

2. **Inicie o servidor FastAPI**
   ```bash
   python run.py
   ```
   > Ou diretamente: `uvicorn app.main:app --reload`

3. **Acesse a documentação interativa**
   - [http://localhost:8000/docs](http://localhost:8000/docs)

## Testes

1. **Execute os testes (exemplo com pytest):**
   ```bash
   pytest
   ```
   > Adapte conforme a ferramenta de testes utilizada.

## Estrutura do Projeto

```text
backend/
├── app/
│   ├── main.py         # Ponto de entrada FastAPI
│   ├── api/           # Rotas e endpoints
│   ├── models/        # Modelos Pydantic/SQLModel
│   ├── services/      # Lógica de negócio e IA
│   ├── db/            # Configuração e scripts do banco
│   └── ...
├── alembic/           # Migrações do banco
├── requirements.txt
├── README.md
├── .env
├── docker-compose.yml
├── init.sql           # Script de inicialização do banco
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

### Desenvolvimento

Use o `docker-compose.yml` para subir PostgreSQL e Redis localmente.

### Produção

Sugestão com Docker:

```dockerfile
# Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Para deploy completo, considere usar Docker Compose ou Kubernetes com volumes persistentes.

## Referências

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [SQLModel](https://sqlmodel.tiangolo.com/)
- [pgvector](https://github.com/pgvector/pgvector)
- [Sentence Transformers](https://www.sbert.net/)
- [Uvicorn](https://www.uvicorn.org/)
- [Alembic](https://alembic.sqlalchemy.org/)

---

## Contribuição

Sinta-se à vontade para abrir issues ou pull requests. Para mudanças significativas, abra uma issue primeiro para discutir.

## Licença

Este projeto está sob a licença MIT.
