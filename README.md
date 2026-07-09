# Personal LLM Assistant v2.0

生产级 RAG 知识库问答系统——混合检索（关键词+语义） + Query Rewriting（多策略） + 多轮对话 + 流式输出。

## Retrieval Performance

在 32 份数据挖掘课程 PDF（2438 chunks）上，400 条 LLM 生成 query 的 Ablation Study：

| 配置 | Hit@5 | MRR | 说明 |
|---|---|---|---|
| B0: 纯 BM25 (关键词) | 0.1525 | 0.1093 | 精确匹配强，语义弱 |
| B3: BM25 + Dense (向量) RRF 融合 | 0.6550 | 0.3742 | 关键词+语义互补，**+50pp** |
| B4: + CrossEncoder 精排 | **0.8200** | **0.7030** | 深度语义重排，**+16pp** |

> 从纯关键词检索到完整管道，Hit@5 提升 **5.4 倍**（15% → 82%），MRR 提升 **6.4 倍**。

### Query Rewriting 评估

18 条手工标注多轮对话样本，代词指代消解准确率：**100%**。

```
Q: "它有什么优缺点？"（上文讨论决策树）
RW: [
  "决策树的优缺点",
  "决策树的优势与劣势",
  "决策树 优点 缺点 适用场景 局限性"
]
```

## Architecture

```
User Message → LangGraph StateGraph
  ├─ rewrite : Multi-Query改写 (指代消解 + 多角度变体 + 查询扩展 + 子问题分解)
  ├─ retrieve: BM25(关键词) + Dense(语义向量) → RRF 融合去重
  ├─ rerank  : CrossEncoder 深度语义精排 (BGE-Reranker-v2-m3, GPU)
  └─ respond : LLM 流式生成 (DeepSeek/通义/智谱)
        ↓
  FastAPI SSE → 会话持久化 (SQLite)
```

### 混合检索详解

```
用户查询 "什么是决策树？"
        │
   ┌────┴────┐
   ▼         ▼
 BM25       Dense (通义 Embedding → ChromaDB)
(精确匹配)  (语义相似)
   │         │
   │    找到含"分类算法/树形模型"的文档
   │    (不含"决策树"关键词但语义相关)
   │         │
 找到含"决策树"的文档 (精确命中)
   │         │
   └────┬────┘
        ▼
   RRF 融合去重 → Top 20
        │
        ▼
   CrossEncoder 精排 → Top 5 → 送 LLM
```

## Quick Start

```bash
# 1. 安装依赖
pip install -e .

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env:
#   DEEPSEEK_API_KEY=sk-xxx  (对话模型)
#   TONGYI_API_KEY=sk-xxx    (Embedding)

# 3. 导入文档
python scripts/ingest_docs.py

# 4. 启动服务
uvicorn app.api.main:app --reload --port 8000

# 5. 打开 http://localhost:8000/docs
```

## API

| Method | Path | Description |
|---|---|---|
| POST | `/chat` | SSE streaming chat（自动创建会话） |
| POST | `/chat/{id}` | 指定会话发送消息 |
| GET | `/conversations` | 会话列表 |
| GET | `/conversations/{id}` | 会话详情（含历史消息） |
| DELETE | `/conversations/{id}` | 删除会话 |
| POST | `/documents/upload` | 上传文档 |
| GET | `/documents` | 已导入文档列表 |
| GET | `/health` | 健康检查 |
| GET | `/stats` | 系统统计 |

SSE 响应示例：

```
event: meta
data: {"conversation_id":"abc123","rewritten_query":"决策树的优缺点"}

event: delta
data: {"delta":"决策树的主要优点包括：1. 易于理解和解释..."}

event: done
data: [DONE]
```

## Evaluation

```bash
# 生成评估数据集 (LLM 辅助)
python eval/generate_dataset.py --sample 200

# 纯 BM25 baseline (秒级)
python eval/run_quick_eval.py

# 完整 pipeline (ingestion + BM25 + Hybrid + CrossEncoder, GPU)
python eval/run_full_gpu.py

# Query Rewriting 专项评估
python eval/eval_rewrite.py
```

## Directory Structure

```
app/
├── api/          # FastAPI + SSE + 路由
├── core/
│   ├── llm/      # 通义/DeepSeek/智谱 adapter
│   ├── retrieval/# BM25 + ChromaDB + RRF + CrossEncoder
│   ├── graph/    # LangGraph 编排 (4 nodes)
│   └── document/ # PDF/Word 解析 + Parent-Child Chunking
├── db/           # 会话/用户/文档 SQLite 持久化
eval/             # 评估脚本 + 数据集 + 结果
scripts/          # 批量文档导入
tests/            # 17 单元测试
legacy/           # v1 demo 归档
```

## Tech Stack

| Layer | Component |
|---|---|
| **LLM** | DeepSeek-Chat / 通义 Qwen / 智谱 GLM |
| **Embedding** | 通义 text-embedding-v3 (API) |
| **关键词检索** | BM25 (rank-bm25) |
| **向量检索** | ChromaDB (cosine similarity) |
| **融合算法** | RRF (Reciprocal Rank Fusion) |
| **精排模型** | BAAI/bge-reranker-v2-m3 (CrossEncoder, CUDA) |
| **Query Rewriting** | Multi-Query 多策略改写 (指代消解 + 变体 + 扩展 + 子问题分解) |
| **编排** | LangGraph + InMemorySaver |
| **Chunking** | Parent-Child (200/800 tokens) |
| **Web** | FastAPI + SSE streaming |
| **会话持久化** | SQLite (aiosqlite) |
