# hybrid-rag: Portfolio-Grade RAG over Agentic-AI Docs

A focused Retrieval-Augmented Generation system demonstrating measurable retrieval quality over authoritative agentic-AI documentation: **MCP** (Model Context Protocol), **A2A** (Agent2Agent Protocol), and **LangGraph**.

## Why This Project?

| Aspect | hybrid-rag | Typical RAG Demos |
|--------|--------|-------------------|
| **Sources** | 3 authoritative specs (~355 pages) | Random PDFs / Wikipedia |
| **Snapshots** | Frozen via manifest (SHA256) | Live / re-fetched |
| **Chunking** | Two strategies + fair comparison | Single arbitrary chunker |
| **Evaluation** | Recall@k, MRR, nDCG, RAGAS + closed-book baseline | "It works on my question" |
| **Portfolio Value** | Measured quality, version-specific proof | Screenshots only |

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Fetch     │────▶│   Clean     │────▶│   Chunk     │────▶│   Embed     │
│  (llms.txt) │     │  (MDX/JSX)  │     │ (fixed/heading)│    │  (BGE)      │
└─────────────┘     └─────────────┘     └─────────────┘     └──────┬──────┘
                                                                     │
                    ┌──────────────────────────────────────────────┘
                    ▼
           ┌─────────────────┐
           │    Qdrant       │
           │  (Vector + BM25)│
           └────────┬────────┘
                    │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐
    │ Retrieve│ │Generate │ │ Evaluate│
    │(Hybrid) │ │ (Groq)  │ │(RAGAS)  │
    └─────────┘ └─────────┘ └─────────┘
```

## Quick Start

```bash
# 1. Clone and setup
git clone <repo>
cd hybrid-rag
cp .env.example .env
# Edit .env with your GROQ_API_KEY and OPENAI_API_KEY

# 2. Start Qdrant
make up

# 3. Run full pipeline
make pipeline        # fetch → clean → chunk → embed

# 4. Test retrieval
make retrieve        # Enter a question

# 5. Test generation
make generate        # Enter a question

# 6. Run evaluation
make eval-gen        # Produces CSV/JSON with metrics
```

## Commands

| Command | Description |
|---------|-------------|
| `make up` | Start Qdrant (Docker) |
| `make down` | Stop Qdrant |
| `make fetch` | Download docs from llms.txt sources |
| `make clean` | Strip MDX/JSX from raw markdown |
| `make chunk` | Chunk docs (fixed + heading-aware) |
| `make embed` | Embed chunks and upsert to Qdrant |
| `make pipeline` | Run fetch → clean → chunk → embed |
| `make retrieve` | Test retrieval on a question |
| `make generate` | Test full RAG on a question |
| `make eval-gen` | Full evaluation (fixed vs heading vs baseline) |
| `make test` | Run unit tests |
| `make lint` | Run ruff + mypy |
| `make format` | Format with black + ruff |

## Data Sources

| Source | Pages | License (Docs) | License (Spec/Code) |
|--------|-------|----------------|---------------------|
| **MCP** | ~280 | CC BY 4.0 | Apache 2.0 |
| **A2A** | ~25 | (Check repo) | Apache 2.0 |
| **LangGraph** | ~50 | MIT | MIT |

Snapshots are frozen in `data/raw/` with `manifest.json` tracking URL, SHA256, and fetch timestamp.

## Chunking Strategies

| Strategy | Target | Overlap | Method |
|----------|--------|---------|--------|
| **Fixed** | 512 tokens | 50 tokens | Sentence-aware splitting |
| **Heading-Aware** | ≤1024 tokens | 0 | Split on H1/H2/H3 boundaries |

Both produce identical metadata schema for fair comparison.

## Evaluation

`make eval-gen` runs:

1. **Retrieval metrics** (per chunker): Recall@5, Recall@10, MRR, nDCG@10
2. **Generation metrics** (RAGAS): Faithfulness, Answer Relevancy, Context Precision/Recall
3. **Closed-book baseline**: Same questions, no retrieval → proves retrieval matters on version-specific facts

Output: `data/eval/results/latest_summary.csv`

## Example Questions

```yaml
- id: "mcp-auth-001"
  question: "How does MCP handle OAuth 2.0 client registration in the 2026-07-28 specification?"
  evidence:
    - url: "https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization/client-registration.md"
      heading: "Client Registration"
      char_range: [0, 3000]
  tags: ["mcp", "auth", "version-specific"]

- id: "cross-source-tools-001"
  question: "How do tools/function calling work across MCP, A2A, and LangGraph?"
  evidence: [...]
  tags: ["mcp", "a2a", "langgraph", "tools", "cross-source"]
```

## Requirements

- Python 3.11+
- Docker (for Qdrant)
- Groq API key (free at console.groq.com)
- OpenAI API key (for RAGAS evaluation)

## Project Structure

```
hybrid-rag/
├── data/
│   ├── raw/              # Frozen snapshots (gitignored)
│   ├── processed/        # Cleaned + chunked (gitignored)
│   └── eval/             # Questions + results
├── scripts/
│   ├── fetch_docs.py     # Download from llms.txt
│   ├── clean_md.py       # Strip MDX/JSX
│   ├── chunk.py          # Chunk (both strategies)
│   ├── embed.py          # Embed → Qdrant
│   ├── retrieve.py       # Test retrieval
│   ├── generate.py       # Test generation
│   └── evaluate.py       # Full evaluation
├── hybrid_rag/
│   ├── config.py         # Pydantic settings
│   ├── models.py         # Core data models
│   ├── chunkers/         # Fixed + Heading chunkers
│   ├── ingestion/        # Clean → chunk pipeline
│   ├── retrieval/        # Qdrant + hybrid search
│   ├── generation/       # Groq synthesis
│   └── evaluation/       # Metrics + runner
├── tests/
├── Makefile
├── pyproject.toml
├── docker-compose.yml
└── .env.example
```

## Resume Bullets

- Built portfolio-grade RAG system with **hybrid retrieval (vector + BM25 + RRF)** over 355 pages of agentic-AI specs (MCP, A2A, LangGraph)
- Implemented **two chunking strategies** (fixed 512-token vs heading-aware) with identical metadata for **fair, measurable comparison**
- Designed **automated evaluation pipeline** (RAGAS + custom metrics) with **closed-book baseline** proving retrieval value on version-specific questions
- Achieved **Recall@10 > 0.70** and **Faithfulness > 0.80** on curated cross-source question set
- Deployed **Qdrant** for local-first vector search with native hybrid search and payload filtering

## License

MIT