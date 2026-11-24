# Certification mentor

A practice exam for Anthropic's Claude Certified Architect – Foundations (CCAR-F). It retrieves study material from a Postgres database and asks one multiple-choice question at a time.

Requires Python 3.14 or newer, and Postgres with the `pgvector` extension.

## Environment

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-openai-api-key
DATABASE_USER=your-database-user
DATABASE_PASSWORD=your-database-password
DATABASE_HOST=your-database-host
DATABASE_PORT=your-database-port
DATABASE_NAME=your-database-name
```

`OPENAI_API_KEY` is sent to the OpenAI API. The `DATABASE_*` variables are the Postgres connection.

Start the database before the app:

```bash
docker run -it \
    --name mentor \
    -e POSTGRES_USER=${DATABASE_USER} \
    -e POSTGRES_PASSWORD=${DATABASE_PASSWORD} \
    -e POSTGRES_DB=${DATABASE_NAME} \
    -v pgvector_data:/var/lib/postgresql/data \
    -p ${DATABASE_PORT}:5432 \
    pgvector/pgvector:pg17
```

If the container already exists, start it with `docker start -ai mentor`.

## Configuration

Settings live in `src/config/config.json`.

`logging.level` is a number: `0` debug, `1` info, `2` warning, `3` error, `4` critical.

`ingestion` controls the first load. `enabled` turns the crawl on. `timeout` is the page fetch limit in seconds. `sources_path` is the list of pages to fetch. `memory.type` selects the store (`vector_db` writes embeddings to Postgres). `memory.embedding_model` is the sentence-transformers model used for those embeddings. `memory.url` is the Postgres URL; `{user}`, `{password}`, `{host}`, `{port}`, and `{db_name}` are filled from the `.env` file.

`categories` is the exam blueprint. Each entry has a `tag` (the value stored on each document and used to filter search), a `description`, a `weight`, and `amount_of_questions`.

## Run with uv

```bash
uv sync
```

## Run with pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

## Start the exam

The first start crawls the pages in `sources_path`, creates the `pgvector` extension and the `documents` table, stores the pages, and sets `ingestion.enabled` to `false`. Later starts skip that step. They expect the table and rows to already be in Postgres, and they do not create them again. A later start against an empty database gets past startup and fails on the first search. Set `ingestion.enabled` back to `true` to load the pages again.

With uv:

```bash
uv run certification-mentor
```

With pip, from the activated virtual environment:

```bash
certification-mentor
```
