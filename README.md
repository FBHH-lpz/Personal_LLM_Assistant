# Personal LLM Assistant v2.0

生产级 RAG 知识库问答系统——混合检索（关键词+语义+图表） + Multi-Query Rewriting + 多轮对话 + 流式输出。

## Retrieval Performance

在 32 份数据挖掘课程 PDF（2438 text + 429 image chunks）上：

| 配置 | Hit@5 | MRR | 说明 |
|---|---|---|---|
| B0: 纯 BM25 (关键词) | 0.1525 | 0.1093 | 精确匹配强，语义弱 |
| B3: BM25 + Dense + Image 三路 RRF | 0.6550 | 0.3742 | 关键词+语义+图表互补 |
| B4: + CrossEncoder 精排 | **0.8200** | **0.7030** | 深度语义重排 |

> 从纯关键词到完整管道，Hit@5 提升 **5.4 倍**（15% → 82%），MRR 提升 **6.4 倍**。

### Query Rewriting 评估

18 条手工标注多轮对话样本，代词指代消解准确率：**100%**。支持 Multi-Query 多角度改写 + 查询扩展 + 子问题分解。

## Architecture

```
User Message → LangGraph StateGraph
  ├─ rewrite : Multi-Query 改写 (指代消解 + 多角度变体 + 查询扩展 + 子问题分解)
  ├─ retrieve: 三路检索 BM25(关键词) + Dense(语义) + Image(图表) → RRF 融合
  ├─ rerank  : CrossEncoder 深度语义精排 (BGE-Reranker-v2-m3, GPU)
  └─ respond : LLM 实时流式生成 (deepseek-v4-pro)
        ↓
  FastAPI SSE → Gradio Chat UI
```

### 文档处理管道

```
PDF → PyMuPDF 文本提取 + 图表检测
         ├─ 文本 → Parent-Child Chunking → [课件N] 标签注入
         └─ 图表 → 页面截图 → Qwen-VL 结构化分析(并发) → 独立 image collection
                        ↓
            {type: "chart", chart_type: "折线图", data_points: [...], key_insights: [...]}
```

## Quick Start

```bash
# 1. 安装依赖
pip install -e .

# 2. 配置 API Key
cp .env.example .env
# DEEPSEEK_API_KEY=sk-xxx   (对话模型)
# TONGYI_API_KEY=sk-xxx     (Embedding + VLM)

# 3. 导入文档（含图表 VLM 分析）
python scripts/ingest_docs.py

# 4. 启动 API
uvicorn app.api.main:app --port 8000

# 5. 启动聊天界面
python app/web_ui.py
# 打开 http://localhost:7860
```

## API

| Method | Path | Description |
|---|---|---|
| POST | `/chat` | 流式聊天（自动创建会话） |
| GET | `/conversations` | 会话列表 |
| GET | `/conversations/{id}` | 会话详情 |
| DELETE | `/conversations/{id}` | 删除会话 |
| POST | `/documents/upload` | 上传文档 |
| GET | `/documents` | 文档列表 |
| GET | `/stats` | 系统统计（含图表数） |

## Evaluation

```bash
python eval/generate_dataset.py    # LLM 辅助生成数据集
python eval/run_quick_eval.py      # BM25 baseline
python eval/run_full_gpu.py        # 完整 ablation
python eval/eval_rewrite.py        # Query Rewriting 专项
```

## Tech Stack

| Layer | Component |
|---|---|
| **LLM** | deepseek-v4-pro / DeepSeek-Chat / 通义 Qwen / 智谱 GLM |
| **Embedding** | 通义 text-embedding-v3 (API) |
| **VLM (图表分析)** | Qwen-VL-Plus (结构化 JSON 输出，并发调用) |
| **关键词检索** | BM25 (rank-bm25，中文分词) |
| **向量检索** | ChromaDB（文本 + 图像双 collection） |
| **融合** | RRF 三路融合 (BM25 + Dense + Image) |
| **精排** | BAAI/bge-reranker-v2-m3 (CrossEncoder, CUDA) |
| **Query Rewriting** | Multi-Query + 查询扩展 + 子问题分解 |
| **编排** | LangGraph StateGraph |
| **Chunking** | Parent-Child (200/800 tokens) |
| **Web** | FastAPI + SSE 真流式 |
| **前端** | Gradio ChatInterface |
| **会话** | SQLite 持久化 |
