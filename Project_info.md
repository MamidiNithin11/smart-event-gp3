# Smart Event Management System

## 1. Project at a glance

Smart Event Management System is a full-stack web application for publishing
and discovering events, managing venues and registrations, maintaining
participant and administrative records, collecting post-event feedback, and
answering event or policy questions through an AI assistant.

The application consists of:

- A React single-page application built with Vite.
- A Python REST API built with FastAPI.
- PostgreSQL persistence through SQLAlchemy and Alembic.
- Optional Redis-backed caching.
- An LLM-powered, LangGraph-based assistant with role-scoped business tools.
- A document ingestion and hybrid-search Retrieval-Augmented Generation (RAG)
  pipeline backed by PostgreSQL and pgvector.
- Docker Compose configuration for running the frontend, backend, and Redis.

The project is suitable for development and demonstration. A production
deployment must provide correctly managed secrets, a secured PostgreSQL
service with required extensions, an LLM provider key, valid CORS settings,
and appropriate operational controls.

## 2. Functional goals and users

The system provides a single interface for managing the event lifecycle:

1. Administrators create and maintain event and venue information.
2. The public browses published events and views event details.
3. Participants create accounts, sign in, register for events, and manage
   their registrations.
4. Administrators oversee users, events, venues, registrations, payments,
   feedback, and AI activity.
5. Participants can submit one rating and comment for an event after it has
   completed.
6. Users can ask the AI assistant questions. Depending on identity and role,
   the assistant can search system data, retrieve policy knowledge, and call
   authorized business operations.

### Application roles

| Role | Intended access |
|---|---|
| `GUEST` | Public event discovery, venue lookup/availability, policy search, and guest AI chat. |
| `USER` | Guest capabilities plus their own profile-related and registration workflows, personal AI chat, and post-event feedback. |
| `ADMIN` | Administrative event, venue, registration, user, payment, feedback, knowledge-base, dashboard, and agent-monitoring workflows. |
| `MAIN_ADMIN` | Admin capabilities plus privileged role-management and cache-clearing operations. |

Authorization is enforced by backend dependencies and service/tool permission
checks; hiding a page in the frontend is not the security boundary.

## 3. Feature inventory

### Public experience

- Landing page and informational About page.
- Browse published events and open event details.
- Browse completed events and publicly visible feedback for completed events.
- Register and log in.
- Responsive navigation, reusable loading/error/empty states, dialogs, badges,
  notifications, and event cards.

### Participant experience

- Authenticated dashboard and profile pages.
- View personal event registrations.
- Register for an event and cancel a personal registration.
- Submit feedback once an event has completed, on the participant's own
  behalf.
- Use the AI assistant and manage authenticated chat sessions.

### Administration

- Dashboard summary and cache status.
- Create, update, publish/manage, and delete event records.
- Manage venues, registrations, users, feedback, and payment records.
- Manage role assignments with `MAIN_ADMIN` restrictions.
- Upload and ingest knowledge documents, inspect index status, and delete
  indexed documents.
- Inspect agent runs and their recorded tool calls.
- Access an admin AI-assistant page and activity view.

### AI assistant

- Natural-language chat with optional guest authentication.
- Persistent conversation sessions and message history for authenticated users;
  guest chat sessions are supported by the backend model.
- Role-filtered tools for event search/details, venue search/availability,
  registration actions, administrative operations, and policy search.
- Bounded tool-use loops and an LLM provider router with configured fallback
  models.
- Recorded agent-run and tool-call metadata for administrative review.
- Optional Server-Sent Events (SSE) chat response endpoint.

## 4. Architecture

```text
Browser
  |
  | React SPA (Vite; /api requests)
  v
FastAPI application
  |-- Authentication and authorization
  |-- Event, venue, registration, payment, feedback, and user routers
  |-- Service layer and SQLAlchemy ORM
  |-- AI chat endpoint -> LangGraph agent -> role-authorized tools
  |                                      |-> business services -> PostgreSQL
  |                                      `-> hybrid RAG -> pgvector / full-text search
  |-- Optional Redis cache
  v
PostgreSQL (application records, agent records, knowledge documents/chunks)

LLM / embeddings: OpenRouter-compatible API configured through environment.
```

### Backend request path

1. FastAPI parses and validates requests through Pydantic schemas.
2. Dependencies resolve a database session and, where required, the current
   user from a bearer token.
3. Routers perform endpoint-level access checks and delegate business
   operations to service modules.
4. SQLAlchemy persists/query data in PostgreSQL.
5. API response models serialize results.

The backend is organized into `api/` for auth/admin/chat APIs, `routers/` for
domain REST endpoints, `services/` for business logic, `models/` for ORM
entities, `schemas/` for API validation/serialization, `core/` for
configuration/security/permissions/cache utilities, and `db/` for engine and
session management.

### Frontend request path

The React application uses React Router layouts and route guards. Service
modules call a shared Fetch-based API client, which sets JSON headers, adds a
stored bearer token when available, applies a 30-second default timeout,
normalizes FastAPI errors, and dispatches an auth-expired event on HTTP 401.
The default API base is `/api`; it can be overridden using `VITE_API_URL`.

## 5. Technology stack

| Area | Technologies |
|---|---|
| Frontend | React 19, JavaScript/JSX, Vite, React Router, CSS |
| UI and interaction | Lucide React, Motion, Sonner, reusable application components |
| 3D/visual components | Three.js, React Three Fiber, Drei |
| Markdown chat rendering | `react-markdown`, `remark-gfm` |
| Backend | Python 3.11+, FastAPI, Uvicorn, Pydantic v2, Pydantic Settings |
| Persistence | PostgreSQL, SQLAlchemy 2, psycopg2, Alembic |
| Authentication | JWT (`python-jose`), Passlib/Bcrypt password hashing |
| AI orchestration | LangGraph, LangChain Core, LangChain OpenAI integration, OpenAI-compatible client |
| AI provider | OpenRouter-compatible chat and embedding endpoints |
| Vector and lexical search | pgvector, PostgreSQL `tsvector` / full-text search |
| Cache | Redis 7+ and Python Redis client |
| Document conversion | MarkItDown for PDF, DOCX, PPTX, and XLSX conversion; text formats handled directly |
| Packaging and development | `uv` for Python; npm for frontend |
| Deployment | Docker, Docker Compose, Nginx frontend image |
| Tests / lint | pytest, pytest-asyncio, HTTPX, Oxlint |

## 6. Repository layout

```text
Smart-Event-Management/
├── README.md
├── Project_info.md
├── SETUP_INSTRUCTIONS.md
├── docker-compose.yml
├── backend/
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── alembic.ini
│   ├── alembic/
│   │   ├── env.py
│   │   └── versions/                 # Schema and index migrations
│   ├── app/
│   │   ├── main.py                   # FastAPI app and router registration
│   │   ├── api/                      # Authentication, admin, chat
│   │   ├── agent/                    # LangGraph, state, prompts, memory, nodes
│   │   ├── core/                     # Settings, security, roles, cache, statuses
│   │   ├── db/                       # SQLAlchemy engine/session
│   │   ├── llm/                      # Provider router, config, provider clients
│   │   ├── models/                   # SQLAlchemy models
│   │   ├── observability/            # Run logging and timing
│   │   ├── rag/                      # Conversion, chunking, ingestion, retrieval
│   │   ├── routers/                  # Domain REST API routes
│   │   ├── schemas/                  # Pydantic request/response schemas
│   │   ├── services/                 # Domain business operations
│   │   └── tools/                    # Role-scoped agent tools and context
│   ├── documents/                    # Seed knowledge and policy documents
│   ├── scripts/                      # Knowledge ingestion utility
│   └── tests/                        # Backend unit/API/journey tests
└── frontend2/
    ├── Dockerfile
    ├── nginx.conf
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── App.jsx                   # Route tree and providers
        ├── auth/                     # Auth provider, roles, token storage
        ├── components/               # Common, event, navigation, chat, payment, 3D
        ├── context/                  # Shared AI chat state
        ├── layouts/                  # Public, user, admin layouts
        ├── pages/                    # Public, auth, user, admin pages
        ├── routes/                   # Protected route guards
        ├── services/                 # API and domain service clients
        └── utils/
```

## 7. Frontend routes

| Route | Access / purpose |
|---|---|
| `/` | Public home |
| `/events` | Published event listing |
| `/events/:eventId` | Event details |
| `/completed-events` | Completed events |
| `/about` | Public About page |
| `/login` | Public-only login page |
| `/register` | Public-only account registration |
| `/dashboard` | Authenticated user dashboard |
| `/profile` | Authenticated profile |
| `/my-events` | Participant-only registration view |
| `/admin` | Admin dashboard |
| `/admin/events` | Event administration |
| `/admin/registrations` | Registration administration |
| `/admin/users` | User administration |
| `/admin/venues` | Venue administration |
| `/admin/feedback` | Feedback moderation |
| `/admin/payments` | Payment-record administration |
| `/admin/ai-chat` | Admin AI assistant |
| `/admin/knowledge` | Knowledge-base administration |
| `/admin/agent-activity` | Agent run monitoring |
| `*` | Not-found page |

The global authentication and AI-chat providers wrap the route tree. The AI
assistant drawer is mounted at the application level.

## 8. Backend API reference

All paths below are relative to the backend origin, normally
`http://localhost:8000`. JSON endpoints use the API schemas for validation and
serialization. Endpoints marked authenticated require a bearer access token.

### Health and authentication

| Method | Path | Access | Purpose |
|---|---|---|---|
| `GET` | `/health` | Public | Liveness response `{ "status": "ok" }`. |
| `POST` | `/api/auth/register` | Public | Create a participant account. |
| `POST` | `/api/auth/login` | Public | Validate credentials and return an access token. |
| `GET` | `/api/auth/me` | Authenticated | Return the current user profile. |

### Events — `/api/events`

| Method | Path | Access | Purpose |
|---|---|---|---|
| `GET` | `/` | Public; admin sees all, others see published | List events. |
| `GET` | `/published` | Public | List published events. |
| `GET` | `/completed` | Public | List completed events. |
| `GET` | `/{event_id}` | Public for published; admin may view non-published | Get event details. |
| `POST` | `/` | Admin | Create an event. |
| `PUT` | `/{event_id}` | Admin | Update an event. |
| `DELETE` | `/{event_id}` | Admin | Delete an event; conflicts can return HTTP 409. |

Event fields include title, description, category, optional venue/address,
start and end timestamps, optional registration deadline, capacity, available
seats, creator, status, and ticket price. The database enforces an end time
later than the event start when an end time is supplied. Events may use a
custom address without a venue.

### Registrations — `/api/registrations`

| Method | Path | Access | Purpose |
|---|---|---|---|
| `GET` | `/` | Admin | List all registrations. |
| `GET` | `/{registration_id}` | Authenticated owner or admin by service access rules | Get one registration. |
| `POST` | `/` | Authenticated participant | Register the caller for an event. |
| `GET` | `/user/{user_id}` | Authenticated owner or admin by service access rules | List registrations for a user. |
| `GET` | `/event/{event_id}` | Admin | List registrations for an event. |
| `PUT` | `/{registration_id}/cancel` | Participant owner | Cancel own registration. |

The endpoint rejects an admin registering as a participant, prevents a
participant from registering another user, and validates referenced user and
event records. A database uniqueness constraint prevents duplicate
user/event registration pairs.

### Venues — `/api/venues`

| Method | Path | Access | Purpose |
|---|---|---|---|
| `GET` | `/` | Public | List venues. |
| `GET` | `/{venue_id}` | Public | Get a venue. |
| `POST` | `/` | Admin | Create a venue. |
| `PUT` | `/{venue_id}` | Admin | Update a venue. |
| `DELETE` | `/{venue_id}` | Admin | Delete a venue. |

Venue records include name, address, city, and capacity.

### Users — `/api/users`

| Method | Path | Access | Purpose |
|---|---|---|---|
| `GET` | `/` | Admin | List users. |
| `GET` | `/email/{email}` | Admin | Find a user by email. |
| `GET` | `/{user_id}` | Admin | Get a user. |
| `POST` | `/` | Admin; only `MAIN_ADMIN` may create admin-role accounts | Create a user account. |
| `PATCH` | `/{user_id}/role` | `MAIN_ADMIN` | Change a user's role. |
| `DELETE` | `/{user_id}` | Admin | Delete a user. |

Public registration is provided separately at `/api/auth/register`.

### Payments — `/api/payments`

| Method | Path | Access | Purpose |
|---|---|---|---|
| `GET` | `/` | Admin | List payment records. |
| `GET` | `/{payment_id}` | Admin | Get one payment record. |
| `GET` | `/user/{user_id}` | Admin | List payment records for a user. |
| `GET` | `/event/{event_id}` | Admin | List payment records for an event. |
| `POST` | `/` | Authenticated user for self, or admin | Create a payment record. |
| `PUT` | `/{payment_id}/status` | Admin | Update payment status. |

Payment records contain user/event references, amount, currency, status,
optional transaction ID, and payment timestamp. The implementation documents
record management; it does not, by these routes, constitute a payment-gateway
checkout integration.

### Feedback — `/api/feedback`

| Method | Path | Access | Purpose |
|---|---|---|---|
| `GET` | `/` | Admin | List all feedback. |
| `GET` | `/public/completed` | Public | List public feedback for completed events. |
| `GET` | `/event/{event_id}` | Admin | List feedback for an event. |
| `GET` | `/user/{user_id}` | Authenticated owner or admin by service access rules | List feedback by user. |
| `POST` | `/` | Authenticated participant, self only | Submit a rating and optional comment. |
| `DELETE` | `/{feedback_id}` | Admin | Delete feedback. |

Feedback is restricted to completed events and one entry per participant per
event. The database constrains ratings to the inclusive range 1–5 and enforces
the unique participant/event pair.

### Admin — `/api/admin`

| Method | Path | Access | Purpose |
|---|---|---|---|
| `GET` | `/dashboard` | Admin | Return administrative dashboard data/metrics. |
| `GET` | `/cache/status` | Admin | Return Redis cache health/statistics. |
| `DELETE` | `/cache/clear` | `MAIN_ADMIN` | Clear Redis cache entries. |

### AI chat and operations

| Method | Path | Access | Purpose |
|---|---|---|---|
| `POST` | `/api/chat` | Guest or authenticated | Run one agent request and return reply/metadata. |
| `POST` | `/api/chat/stream` | Guest or authenticated | Return the completed agent reply as SSE character events followed by metadata. |
| `GET` | `/api/chat/sessions` | Authenticated | List the caller's sessions. |
| `GET` | `/api/chat/sessions/{session_id}` | Authenticated session owner | Read session messages. |
| `DELETE` | `/api/chat/sessions/{session_id}` | Authenticated session owner | Delete one session. |
| `DELETE` | `/api/chat/sessions` | Authenticated | Delete the supplied session IDs owned by the caller. |
| `GET` | `/api/admin/agent-activity` | Admin | List recent agent runs (request limit is capped at 200). |
| `GET` | `/api/admin/agent-activity/{run_id}` | Admin | Read an agent run and associated tool-call details. |
| `POST` | `/api/admin/knowledge/upload` | Admin | Upload and index a supported knowledge document. |
| `POST` | `/api/admin/knowledge/ingest` | Admin | Ingest/re-ingest configured knowledge documents. |
| `DELETE` | `/api/admin/knowledge/documents/{filename}` | Admin | Remove a knowledge document and its indexed chunks. |
| `GET` | `/api/admin/knowledge/status` | Admin | Inspect knowledge-index status. |

The regular chat response includes the reply, session and run IDs, detected
intent, tools used, sources, latency, and status. Pass a returned `session_id`
to continue that conversation. The stream endpoint currently runs the agent
before yielding reply characters; it is not incremental token generation from
the model provider.

When `ENABLE_API_DOCS=true`, FastAPI exposes Swagger UI at `/docs` and ReDoc at
`/redoc`. Docker Compose sets API docs off for the backend container.

## 9. Data model

Application IDs are PostgreSQL UUIDs. Timestamps are timezone-aware where
defined, and creation timestamps are generally assigned by the database.

| Entity/table | Main data | Relationships and constraints |
|---|---|---|
| `users` | Name, unique email, password hash, role, created timestamp | Referenced by events, registrations, payments, feedback, and agent records. |
| `venues` | Name, address, city, capacity | Optional event venue; indexed city/capacity. |
| `events` | Title, description, category, venue/address, schedule, deadline, capacity/seats, creator, status, ticket price | Optional venue; creator references user; indexed date/status/venue/creator; end must follow start. |
| `registrations` | User, event, registration/creation timestamps, status | Foreign keys to user/event; unique `(user_id, event_id)`. |
| `payments` | User, event, amount, currency, status, transaction ID, payment date | Foreign keys to user/event; transaction ID unique when present. |
| `feedback` | User, event, rating, comment, creation timestamp | Foreign keys to user/event; unique `(user_id, event_id)`; rating 1–5. |
| `knowledge_documents` | Filename, title, inferred type, content hash, chunk count, timestamps | One document entry per filename. |
| `knowledge_chunks` | Legacy 384-dimensional embedding and text-search fields | Retained for compatibility; current ingestion/search uses the NVIDIA-index table. |
| `knowledge_chunks_nvidia` | Text chunk, source, type, content hash, model ID, 2048-dimensional embedding, full-text vector, JSON metadata | Foreign key to knowledge document; unique document/chunk index; vector/full-text indexes are migration-managed. |
| `agent_sessions` | Optional owner, title, created/last-active timestamps | Owner may be null for guest sessions. |
| `chat_messages` | Session, role, content, created timestamp | Session foreign key with cascading delete. |
| `agent_runs` | User request, intent, final response, loop count, status/error, latency, timestamps | References session and optional user. |
| `tool_calls` | Tool name/input/output, status/error, latency, sequence | References an agent run with cascading delete. |

Alembic owns schema creation and evolution. Run migrations against the target
database instead of relying on automatic application startup table creation.
The migration history includes initial domain tables, indexes/optimization,
agent and RAG tables, embedding-index changes, guest sessions, event end time,
and custom event address support.

## 10. Authentication and authorization

- Passwords are hashed and verified using Passlib's Bcrypt context.
- Login returns a signed JWT access token using the configured signing
  algorithm (default `HS256`) and expiration duration.
- Authenticated API requests provide the JWT access token in the HTTP bearer
  authorization header.
- The JWT subject identifies the user; backend authentication reloads that
  user from PostgreSQL.
- Role constants are `USER`, `ADMIN`, and `MAIN_ADMIN`; `GUEST` is used as an
  AI/tool context for unauthenticated chat.
- Admin and main-admin checks are explicit backend dependencies.
- Participant registration and feedback routes check that the caller acts
  only for their own user ID.
- Agent tools are selected by role. Tool context carries database-derived
  identity/role rather than trusting identity values supplied in the prompt.
- CORS origins, API docs exposure, and production secret checks are
  configurable. Production mode rejects a short or known-insecure signing
  secret.

The frontend stores and attaches a bearer token through its auth-storage
module, restores the profile using `/api/auth/me`, and clears the local
session when an unauthorized response triggers the auth-expired event.

## 11. AI agent and tool layer

### Agent execution

The LangGraph graph uses state nodes for memory loading, selected fast-path
queries, model reasoning, tool execution, structured event creation, and final
response assembly. The agent:

1. Resolves an authenticated actor and role where present.
2. Creates or loads a conversation session and logs an agent run.
3. Loads a bounded conversation history window (configurable).
4. Gives the LLM role-appropriate tool definitions.
5. Executes authorized tools through a context-aware tool registry.
6. Repeats reasoning/tool execution only up to the configured tool-loop cap.
7. Saves final response, run metadata, and tool-call records.

### Tool catalog

Guest-capable tools include event search/details, venue search and availability
checks, and policy retrieval. Participant tools add registration, cancellation,
and personal-registration lookup. Admin tools include event management,
registration inspection/counts, venue management, user management, role
updates, and agent activity/run inspection. The role-management tool requires
the stronger main-admin policy where applicable.

Tools call application services rather than issuing model-authored arbitrary
SQL. Sensitive tool arguments are redacted in tool-call logging where
configured by the agent node.

### LLM provider routing

The backend uses an OpenAI-compatible provider interface configured for
OpenRouter. A primary chat model and ordered fallback model IDs can be set in
the environment. Timeouts, retry limits, total operation budget, and
temperature are configurable. Chat model configuration is required rather
than hard-coded to a production credential.

## 12. RAG and knowledge-base pipeline

### Ingestion

1. Read files from `KNOWLEDGE_DOCS_DIR` or accept an admin upload.
2. Validate the filename and supported extension.
3. Convert supported files to text; plain text/Markdown are read directly and
   office/PDF/web/data formats are converted with MarkItDown.
4. Infer a document type from recognized policy/FAQ filenames; otherwise use
   `general`.
5. Compute a content hash and skip unchanged, already indexed documents unless
   forced.
6. Split text by Markdown headings and into overlapping bounded chunks.
7. Extract chunk metadata and build embedding text that includes document and
   section context.
8. Request OpenRouter embeddings and persist chunks to
   `knowledge_chunks_nvidia`.
9. Populate PostgreSQL full-text search vectors and invalidate policy cache.

Supported extensions are `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.html`, `.htm`,
`.csv`, `.json`, `.xml`, `.md`, `.txt`, and `.markdown`.

The active embedding index is `nvidia`, using a 2048-dimensional embedding
model configured by `OPENROUTER_EMBEDDING_MODEL`. Embedding dimension is
validated as 2048. PostgreSQL must have the `vector` extension and support the
pgvector index types used by the migrations.

### Retrieval and generation

- **Dense retrieval:** query embeddings and vector-distance search against the
  active pgvector index.
- **Sparse retrieval:** PostgreSQL English full-text search with
  `websearch_to_tsquery` and `ts_rank_cd`.
- **Fusion:** reciprocal-rank fusion combines dense and sparse candidate
  rankings.
- **Optional reranking:** disabled by default; can use an LLM-based reranker or
  an optional local sentence-transformer cross-encoder.
- **Generation:** the LLM answers using selected chunks and returns grounding
  and source metadata; hallucination checking/retry controls are configurable.

If one retrieval stage fails, the pipeline can use results from the other
stage. If all available search stages fail, retrieval reports an error rather
than silently claiming a successful grounded result.

### Included knowledge materials

The backend documents folder includes policy/FAQ materials such as
`event_policy.md`, `registration_policy.md`, `cancellation_policy.md`,
`venue_policy.md`, `attendance_policy.md`, and `faq.md`, alongside additional
text and office documents. The ingestion command indexes supported files in
that directory; which documents are available to retrieval depends on the
current database index.

### AI observability

Agent run records retain request, intent, response, loop count, status,
latency, and error metadata. Tool-call records retain tool name, inputs/outputs,
status, sequence, and latency. Timing instrumentation covers agent, LLM,
database, and RAG stages.

## 13. Cache and performance

- Redis caching is controlled by `ENABLE_REDIS_CACHE`, `REDIS_URL`, and a
  configurable TTL.
- Docker Compose supplies Redis 7 and waits for its health check before
  starting the backend.
- Cache observability and cache invalidation endpoints are admin-only, with
  global cache clearing restricted to `MAIN_ADMIN`.
- PostgreSQL models and migrations define indexes for common event date/status,
  venue, registration, payment, and feedback access paths.
- RAG uses vector and GIN full-text indexes as configured by migrations.
- Backend database sessions are managed per request and engine connections
  use pre-ping/recycling options.

Redis is an optimization, not the system of record. PostgreSQL is required for
application persistence and AI/RAG data.

## 14. Configuration

Use `backend/.env.example` as the environment-variable reference. Create a
private `backend/.env` for local work; never commit credentials.

| Variable | Purpose |
|---|---|
| `APP_ENV` | Environment mode; `production` enforces stronger secret validation. |
| `DATABASE_URL` | PostgreSQL connection URL. |
| `SECRET_KEY` | JWT signing secret; use a long random value in production. |
| `ALGORITHM` | JWT signing algorithm (example: `HS256`). |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | JWT lifetime. |
| `CORS_ORIGINS` | Comma-separated allowed browser origins. |
| `ENABLE_API_DOCS` | Toggle `/docs` and `/redoc`. |
| `EVENT_DEFAULT_DURATION_HOURS` | Default event-duration setting. |
| `REDIS_URL` | Redis connection URL. |
| `ENABLE_REDIS_CACHE` | Enable/disable Redis-backed caching. |
| `REDIS_CACHE_TTL_SECONDS` | Cache lifetime. |
| `OPENROUTER_API_KEY` | Provider credential for chat and embeddings. |
| `OPENROUTER_BASE_URL` | OpenRouter-compatible API base URL. |
| `OPENROUTER_MODEL` | Primary chat model ID. |
| `OPENROUTER_FALLBACK_MODELS` | Ordered fallback chat model IDs. |
| `LLM_TIMEOUT_SECONDS` | Per-provider request timeout. |
| `LLM_TOTAL_TIMEOUT_SECONDS` | Total budget across provider/fallback attempts. |
| `LLM_MAX_RETRIES` | Provider retry policy. |
| `CHAT_TEMPERATURE` | Chat generation temperature. |
| `OPENROUTER_EMBEDDING_MODEL` | Embedding model ID. |
| `OPENROUTER_EMBEDDING_DIM` | Must be `2048` for the active vector index. |
| `RAG_ACTIVE_EMBEDDING_INDEX` | Must be `nvidia` for the current implementation. |
| `EMBEDDING_MAX_TOKENS`, `EMBEDDING_BATCH_SIZE` | Embedding request sizing. |
| `CHUNK_SIZE_CHARS`, `CHUNK_OVERLAP_CHARS`, `CHUNK_MIN_CHARS` | Chunking controls. |
| `DENSE_TOP_K`, `SPARSE_TOP_K`, `RRF_K`, `FUSED_TOP_K`, `FINAL_TOP_K` | Hybrid retrieval/fusion candidate limits. |
| `ENABLE_RERANKER` | Enable optional reranking. |
| `ENABLE_METADATA_EXTRACTION` | Toggle metadata extraction. |
| `ENABLE_HALLUCINATION_CHECK`, `MAX_GENERATION_RETRIES` | Grounding/retry controls. |
| `AGENT_MAX_TOOL_LOOPS` | Bound the agent tool cycle. |
| `MEMORY_WINDOW_MESSAGES` | Conversation history limit passed into the agent. |
| `KNOWLEDGE_DOCS_DIR` | Knowledge file directory; normally `documents` from backend working directory. |
| `VITE_API_URL` | Optional frontend API base URL; defaults to `/api`. |

Keep the environment file private. The actual values belong in secret
management, not in this documentation.

## 15. Setup and running

### Prerequisites

- Git.
- Docker Desktop and Docker Compose for the recommended full-stack path.
- PostgreSQL with pgvector available for AI/RAG features.
- OpenRouter API key for LLM and embedding features.
- For local backend: Python 3.11+ and `uv`.
- For local frontend: Node.js 18/20+ and npm.
- Redis 7+ for local cache use; Compose provides Redis.

### Configure environment

From the repository root, copy `backend/.env.example` to `backend/.env` and
set at least:

- PostgreSQL `DATABASE_URL`.
- A strong random `SECRET_KEY`.
- `OPENROUTER_API_KEY`.
- `OPENROUTER_MODEL`.
- `CORS_ORIGINS` matching the browser origin.
- For Compose, `REDIS_URL=redis://redis:6379/0`.

Do not place real secrets in source control.

### Docker Compose (recommended)

```powershell
docker compose up --build -d
docker compose ps
docker compose logs -f backend
```

Services and default local URLs:

| Service | URL / port |
|---|---|
| Frontend | `http://localhost` / host port 80 |
| Backend | `http://localhost:8000` |
| Health | `http://localhost:8000/health` |
| Redis | `localhost:6379` |

The frontend waits for a healthy backend, and the backend waits for healthy
Redis. PostgreSQL is external to this Compose file and must be reachable using
`DATABASE_URL`. API docs are disabled in the Compose backend environment.

Stop without deleting persistent Redis data:

```powershell
docker compose down
```

Removing the Redis volume or local images is destructive to that local cache
state; only do so when intentionally cleaning a development environment.

### Local development

Backend, from `backend/`:

```powershell
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend, from `frontend2/` in a separate terminal:

```powershell
npm install
npm run dev
```

The Vite development server runs at `http://localhost:5173` and proxies API
requests to `http://localhost:8000`. Configure CORS to allow the Vite origin.
If Redis is not running locally, set `ENABLE_REDIS_CACHE=false`.

### Database migrations

For a new or updated database, run from `backend/`:

```powershell
uv run alembic upgrade head
```

When using Docker only:

```powershell
docker compose run --rm backend alembic upgrade head
```

The current RAG configuration uses a 2048-dimensional OpenRouter index. After
applying the relevant embedding migration to a database with existing
knowledge documents, re-index them through the admin knowledge-ingestion
endpoint.

### Ingest knowledge files

From `backend/`, the repository utility supports indexing files in the
configured knowledge directory:

```powershell
uv run python scripts/ingest_knowledge.py
uv run python scripts/ingest_knowledge.py --force
uv run python scripts/ingest_knowledge.py --clear
```

Use `--force` when intentionally rebuilding embeddings, and `--clear` only
when intentionally clearing the active index. Confirm indexing using the
admin knowledge status endpoint or the Admin Knowledge Base page.

## 16. Testing and quality checks

Backend test suite is under `backend/tests/` and includes authentication,
password reset, schema validation, custom event-address API behavior,
feedback journey, and user/event journey coverage.

From `backend/`:

```powershell
uv run pytest
```

Frontend scripts in `frontend2/package.json`:

```powershell
npm run lint
npm run build
npm run preview
```

The production frontend build validates Vite compilation and emits static
assets served by the frontend container. API behavior should also be checked
against a migrated PostgreSQL database configured with required extensions.

## 17. Operations and troubleshooting

### Backend health fails

- Check `docker compose logs backend`.
- Verify that PostgreSQL is reachable and `DATABASE_URL` is correct.
- Confirm Alembic migrations were applied.
- Verify that `SECRET_KEY` meets production validation if `APP_ENV=production`.

### Frontend API requests fail

- Check `http://localhost:8000/health`.
- Ensure frontend API base/proxy is correct.
- Ensure `CORS_ORIGINS` contains the exact browser origin, including scheme
  and port.
- Check browser network errors and backend logs.

### AI or RAG requests fail

- Verify the OpenRouter API key, selected chat model, provider availability,
  and timeout configuration.
- Check the active embedding index is `nvidia` and embedding dimension is
  2048.
- Confirm PostgreSQL pgvector extension and migrations are installed.
- Confirm documents have been indexed and inspect
  `/api/admin/knowledge/status`.
- Check agent activity and backend logs for failed runs/tool calls.

### Docker build cannot reach package registries

Check Docker Desktop network/DNS settings and retry the build. The Compose
configuration sets DNS servers for the application containers.

### Ports are occupied

Default host ports are 80, 8000, and 6379. Resolve the conflict or modify the
Compose port mappings before starting.

## 18. Scope boundaries and implementation notes

- Event creation and administration are implemented; the project does not
  imply an external event-ticketing marketplace.
- Payment APIs manage payment records and status. No payment processor or
  card-capture workflow is established by the documented endpoints.
- The system uses bearer JWTs; refresh-token rotation and secure HTTP-only
  cookie delivery are not part of the described auth flow.
- Redis is optional for local work and should not be treated as durable
  business storage.
- The current RAG index is OpenRouter embeddings only (`nvidia`, 2048
  dimensions); the legacy knowledge table remains in the schema for
  compatibility, not as the active retrieval target.
- The SSE chat route emits characters after the agent finishes rather than
  streaming model tokens while generation is in progress.
- API documentation is controlled by configuration and disabled by the
  Compose production environment.
- Frontend route guards improve navigation and user experience; backend role
  checks remain authoritative.

## 19. Useful source locations

| Concern | Source |
|---|---|
| FastAPI startup, CORS, router registration, health endpoint | `backend/app/main.py` |
| JWT, password hashing, bearer authentication | `backend/app/core/security.py` |
| Roles and actor authorization | `backend/app/core/permissions.py` |
| Runtime settings | `backend/app/core/config.py`, `backend/app/core/ai_config.py` |
| Database engine and session | `backend/app/db/database.py` |
| Domain APIs | `backend/app/routers/` |
| Domain business rules | `backend/app/services/` |
| LangGraph agent graph/state/nodes | `backend/app/agent/` |
| Agent tool catalog and implementations | `backend/app/tools/` |
| LLM provider selection | `backend/app/llm/` |
| Knowledge ingestion/retrieval | `backend/app/rag/` |
| Agent and timing observability | `backend/app/observability/` |
| Frontend route tree | `frontend2/src/App.jsx` |
| Frontend API client | `frontend2/src/services/api/client.js` |
| Docker services | `docker-compose.yml` |

## 20. Project background

The repository describes this project as developed collaboratively for the
Infosys Springboard Virtual Internship Program. Its purpose is educational and
professional learning, with an architecture that combines conventional
event-management workflows with an AI agent and document-grounded assistance.
