# Personal LLM Assistant v2.0

生产级 RAG 知识库问答系统，支持混合检索（BM25 + Dense + CrossEncoder）、多轮对话上下文管理（Query Rewriting）、流式输出。

Built with FastAPI + LangGraph + Milvus + BM25 + CrossEncoder.

## Retrieval Performance

在 32 份数据挖掘课程 PDF（2438 chunks）上，400 条 LLM 生成 query 的评估结果：

| Metric | BM25 | +Hybrid(RRF) | +CrossEncoder | 总提升 |
|---|---|---|---|---|
| **Hit@5** | 0.1525 | 0.6550 | **0.8200** | **5.4×** |
| Hit@10 | 0.1950 | 0.8000 | 0.8200 | 4.2× |
| **MRR** | 0.1093 | 0.3742 | **0.7030** | **6.4×** |
| Recall@5 | 0.0584 | 0.2108 | 0.3847 | 6.6× |
| NDCG@5 | 0.0613 | 0.2050 | 0.4535 | 7.4× |

> Hit@5=0.82 意味着 82% 的查询在 top-5 结果中包含了正确答案。完整 ablation 见 `eval/results/full_gpu_eval.json`。

## Architecture

```
User Message → LangGraph StateGraph
  ├─ rewrite:  LLM 指代消解 + 意图补全 (轻量模型)
  ├─ retrieve: BM25(sparse) + Dense(vector) → RRF 融合
  ├─ rerank:   CrossEncoder 精排 (BGE-Reranker-v2-m3, GPU)
  └─ respond:  LLM 流式生成 (DeepSeek/通义/智谱)
       ↓
  FastAPI SSE streaming → Frontend
```

## Quick Start

```bash
# 1. 安装依赖
pip install -e .

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env 填入:
#   DEEPSEEK_API_KEY=sk-xxx  (对话模型)
#   TONGYI_API_KEY=sk-xxx    (Embedding 模型)

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
| POST | `/chat/{id}` | Send message to conversation |
| GET | `/conversations` | List conversations |
| GET | `/conversations/{id}` | Get conversation detail |
| DELETE | `/conversations/{id}` | Delete conversation |
| POST | `/documents/upload` | Upload document for ingestion |
| GET | `/documents` | List ingested documents |
| GET | `/health` | Health check |
| GET | `/stats` | System statistics |

## Evaluation

```bash
# 生成评估数据集 (LLM 辅助)
python eval/generate_dataset.py --sample 200

# 纯 BM25 baseline (秒级)
BM25_INDEX_PATH=data/bm25_index_v2.pkl python eval/run_quick_eval.py

# 完整评估 (ingestion + BM25 + Hybrid + CrossEncoder, GPU 推荐)
python eval/run_full_gpu.py
```

Ablation 数据解读：
- **BM25 → Hybrid(RRF)**: Hit@5 +50pp (15%→66%)，密集向量弥补了关键词匹配的语义鸿沟
- **Hybrid → CrossEncoder**: Hit@5 +16pp (66%→82%)，精排显著提升排名质量（MRR +0.33）
- CrossEncoder 对排名质量（MRR）贡献最大：0.37 → 0.70

## Directory Structure

```
app/
├── api/          # FastAPI routes + middleware
├── core/
│   ├── llm/      # Provider adapters (通义/DeepSeek/智谱)
│   ├── retrieval/# Hybrid search + reranker + evaluator
│   ├── graph/    # LangGraph pipeline (4 nodes)
│   └── document/ # Parser + parent-child chunker + ingestor
├── db/           # SQLAlchemy models + sessions
eval/             # Metrics, dataset gen, run scripts, results
scripts/          # Batch ingestion
tests/            # 17 unit tests passing
legacy/           # Archived v1 demo scripts
```

## Tech Stack

| Layer | Component |
|---|---|
| **LLM** | DeepSeek-Chat / 通义 Qwen / 智谱 GLM |
| **Embedding** | 通义 text-embedding-v3 (API) |
| **Orchestration** | LangGraph + SQLite checkpointing |
| **Vector Store** | Milvus Lite |
| **Sparse Retrieval** | BM25 (rank-bm25) |
| **Reranker** | BAAI/bge-reranker-v2-m3 (CrossEncoder, GPU) |
| **Chunking** | Parent-Child (200/800 tokens) |
| **Web** | FastAPI + SSE streaming |
| **DB** | SQLite (aiosqlite) |
