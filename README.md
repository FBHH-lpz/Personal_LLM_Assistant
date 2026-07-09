# Personal LLM Assistant v2.0

生产级 RAG 知识库问答系统，支持混合检索、多轮对话上下文管理、流式输出。

Built with FastAPI + LangGraph + Milvus + BM25 + CrossEncoder.

## Architecture

```
FastAPI (SSE streaming) → LangGraph (orchestration)
  ├─ rewrite (指代消解 + 意图补全)
  ├─ retrieve (BM25 + Dense 混合检索 → RRF 融合)
  ├─ rerank (CrossEncoder 精排)
  └─ respond (LLM 生成 + 流式输出)
```

## Quick Start

```bash
# 1. 安装依赖
pip install -e .

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env 填入你的 LLM API Key（通义/DeepSeek/智谱）

# 3. 导入文档
python scripts/ingest_docs.py

# 4. 启动服务
uvicorn app.api.main:app --reload --port 8000

# 5. 打开 http://localhost:8000/docs 交互测试
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/chat` | SSE streaming chat |
| GET | `/conversations` | List conversations |
| GET | `/conversations/{id}` | Get conversation detail |
| DELETE | `/conversations/{id}` | Delete conversation |
| POST | `/documents/upload` | Upload document for ingestion |
| GET | `/documents` | List ingested documents |
| GET | `/health` | Health check |
| GET | `/stats` | System statistics |

## Evaluation

```bash
# 生成评估数据集
python eval/generate_dataset.py --sample 200

# 运行评估
python eval/run_eval.py --config hybrid_full --dataset eval/dataset.jsonl

# 对比两个配置
python eval/run_eval.py --compare eval/results/a.json eval/results/b.json
```

## Directory Structure

```
app/
├── api/          # FastAPI routes + middleware
├── core/
│   ├── llm/      # Provider adapters (通义/DeepSeek/智谱)
│   ├── retrieval/# Hybrid search + reranker
│   ├── graph/    # LangGraph pipeline
│   └── document/ # Parser + chunker + ingestor
├── db/           # SQLAlchemy models + sessions
eval/             # Evaluation metrics + dataset generation
scripts/          # Batch ingestion scripts
tests/            # Unit + integration tests
legacy/           # Archived v1 demo scripts
```

## Tech Stack

- **LLM**: 通义千问 / DeepSeek / 智谱 GLM (cloud API)
- **Embedding**: 通义 text-embedding-v3
- **Orchestration**: LangGraph + SQLite checkpointing
- **Vector Store**: Milvus Lite
- **Sparse Retrieval**: BM25 (rank-bm25)
- **Reranker**: BAAI/bge-reranker-v2-m3 (CrossEncoder)
- **Web**: FastAPI + SSE streaming
- **DB**: SQLite (via SQLAlchemy + aiosqlite)
