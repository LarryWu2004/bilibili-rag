# BiliRAG

BiliRAG turns a personal Bilibili favorites folder into a searchable, source-traceable knowledge base.

It synchronizes saved videos, obtains subtitles or speech transcripts, splits content into searchable chunks, stores embeddings in ChromaDB, and uses hybrid retrieval to answer questions with links back to the source videos.

## Pipeline

```text
Bilibili favorites
  -> SQLite metadata and cache
  -> ASR / subtitles / fallback metadata
  -> recursive chunking and embeddings
  -> ChromaDB

Question
  -> route
  -> vector MMR + SQLite keyword recall
  -> weighted RRF fusion
  -> video-level diversity constraint
  -> LLM context
  -> answer with source videos
```

For normal RAG questions, the LLM receives the retrieved `Document.page_content` chunks, not the full text of every video. The response also includes each source video's BV ID, title, and URL. List and summary questions use dedicated database-backed routes when their intent requires reading a collection or cached video content.

## Features

- QR-code login and Bilibili favorites synchronization
- Multi-part video processing
- ASR, subtitle, AI summary, and basic-information fallback
- SQLite cache and persistent ChromaDB index
- Vector retrieval with MMR
- SQLite keyword recall with Chinese n-grams
- Weighted Reciprocal Rank Fusion
- Per-video result limits and source deduplication
- Streaming and non-streaming chat
- Markdown export for original or organized video content

## Stack

- Frontend: Next.js, React, Tailwind CSS
- Backend: FastAPI, Uvicorn
- Persistence: SQLite, SQLAlchemy, aiosqlite
- Retrieval: LangChain, ChromaDB, DashScope Embeddings
- Model services: OpenAI-compatible chat API and DashScope ASR
- Media processing: ffmpeg
- Deployment: Docker Compose

## Run with Docker

Create an environment file and add a DashScope or OpenAI-compatible API key:

```powershell
Copy-Item .env.example .env
docker compose up -d --build
```

Open:

- Web app: http://localhost:3000
- API documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/health

The `data/` and `logs/` directories are mounted to the host. Stop the services with:

```powershell
docker compose down
```

## Configuration

The most relevant settings are:

```env
DASHSCOPE_API_KEY=your-api-key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen3-max
EMBEDDING_MODEL=text-embedding-v4
CHAT_USE_LLM_ROUTER=false

RETRIEVAL_CANDIDATE_K=24
RETRIEVAL_TOP_K=8
RETRIEVAL_MMR_FETCH_K=32
RETRIEVAL_MMR_LAMBDA=0.55
```

Keep `.env` out of version control. Recreate the backend container after changing model or key settings:

```powershell
docker compose up -d --force-recreate backend
```

## API entry points

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/auth/qrcode` | Create a login QR code |
| GET | `/favorites/list` | List favorites folders |
| POST | `/knowledge/folders/sync` | Synchronize folders |
| POST | `/knowledge/build` | Start knowledge-base ingestion |
| GET | `/knowledge/build/status/{task_id}` | Read ingestion progress |
| POST | `/chat/ask` | Non-streaming answer |
| POST | `/chat/ask/stream` | Streaming answer |
| POST | `/chat/search` | Direct chunk retrieval |
| GET | `/knowledge/stats` | Read vector-store statistics |

See `/docs` for request and response schemas.

## Repository layout

```text
app/          FastAPI routes, services, models, and database
frontend/     Next.js client
data/         SQLite and ChromaDB persistence
logs/         Runtime logs
test/         Unit and integration tests
```

## Validation

```powershell
pytest -q
```

## Scope

This project indexes content that the user can access; it does not grant rights to third-party videos, audio, or subtitles. Bilibili interfaces and media URLs can change, and ASR, embedding, and LLM requests may incur usage fees.

## License

Apache-2.0
