# Personal LLM Assistant v2.1

生产级 RAG 知识库问答系统——Docling 版面分析 + 混合检索（BM25+语义+图表） + 表格感知分块 + 多策略 Query Rewriting + 多轮对话 + 流式输出。

## What's New in v2.1

- **Docling 版面分析**：替代 PyMuPDF 纯文本提取，自动识别标题层级、表格结构（TableFormer）、公式、阅读顺序
- **表格感知分块**：Markdown 表格被识别为原子单元，整表不拆散
- **跨页断裂修复**：正则脚本清理 PDF 页边界断词、重复页眉页脚
- **VLM 图文绑定**：图片描述确定性拼入同页父块，同时保留独立图片检索通路
- **三路检索简化**：保留 BM25 + 文本向量 + 图片向量三路 RRF 融合，移除冗余的 Cross-Modal Boost

## Retrieval Performance

在 29 份数据挖掘课程 PDF（1192 页，998 个块）上：

| 配置 | Hit@5 | MRR | 说明 |
|---|---|---|---|
| BM25 (关键词基线) | 0.0734 | 0.0258 | 纯关键词稀疏检索 |
| BM25 + Dense + Image RRF | 0.8853 | 0.4613 | 三路混合融合 |
| + CrossEncoder 精排 | **0.9633** | **0.8644** | BGE-reranker-v2-m3 GPU |

> 从纯 BM25 到完整管道（三路 RRF + CrossEncoder），Hit@5 提升 **13 倍**（7% → 96%），MRR 提升 **33 倍**。

### Query Rewriting 评估

18 条手工标注多轮对话样本，代词指代消解准确率：**100%**。支持多策略改写（指代消解 + 多查询变体 + 查询扩展 + 子问题分解 + 闲聊检测），单个 Prompt 统一调度。

## Architecture

```
用户输入 → LangGraph StateGraph
  ├─ rewrite : 多策略改写 (deepseek-chat)
  ├─ retrieve: 三路并发 (BM25 + Dense Text + Dense Image) → RRF k=60
  ├─ rerank  : CrossEncoder 精排 (BGE-Reranker-v2-m3, GPU optional)
  └─ respond : LLM 流式生成 (deepseek-v4-pro)
        ↓
  FastAPI SSE → Gradio Chat UI
```

### 文档处理管道

```
PDF → Docling 版面分析 (TableFormer + RapidOCR)
        ├─ 结构化 Markdown (标题/表格/列表/公式)
        ├─ 正则跨页修复 (连字符断词/句中断行/重复页眉)
        ├─ 表格感知父子分块 (200/800 tokens, 表格原子化)
        └─ Qwen-VL-Plus 图表分析 (并发) → 拼入父块 + 独立图片索引
```

## Quick Start

```bash
# 1. 安装依赖
pip install -e .

# 2. 配置 API Key
cp .env.example .env
# DEEPSEEK_API_KEY=sk-xxx   (对话 + 查询改写)
# TONGYI_API_KEY=sk-xxx     (Embedding + VLM)

# 3. 导入文档
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
| POST | `/chat` | SSE 流式聊天（自动创建会话） |
| GET | `/conversations` | 会话列表 |
| GET | `/conversations/{id}` | 会话详情 |
| DELETE | `/conversations/{id}` | 删除会话 |
| POST | `/documents/upload` | 上传文档 |
| GET | `/documents` | 文档列表 |
| GET | `/stats` | 系统统计 |

## Evaluation

```bash
python eval/generate_dataset.py      # LLM 生成评估数据集
python eval/run_quick_eval.py        # BM25 基线
python eval/run_retrieval_eval.py    # BM25 + Dense + Image RRF
python eval/eval_rewrite.py          # Query Rewriting 专项评估
```

## Tech Stack

| Layer | Component |
|---|---|
| **版面分析** | Docling (TableFormer + RapidOCR) |
| **LLM** | DeepSeek (deepseek-v4-pro / deepseek-chat) / 通义 Qwen / 智谱 GLM |
| **Embedding** | 通义 text-embedding-v3 (API, 1024d) |
| **VLM (图表)** | Qwen-VL-Plus (结构化 JSON, 并发 Semaphore(5)) |
| **关键词检索** | BM25 (rank-bm25, jieba 中文分词) |
| **向量检索** | ChromaDB (HNSW, cosine, 文本+图像双 collection) |
| **融合** | RRF 三路融合 (k=60, 非参数) |
| **精排** | BAAI/bge-reranker-v2-m3 (CrossEncoder, sentence-transformers) |
| **分块** | Parent-Child (200/800 tokens) + 表格感知 |
| **跨页修复** | 正则脚本（连字符/句中断行/页眉页脚） |
| **Query Rewriting** | 多策略 LLM 改写 (指代消解 + 多查询 + 扩展 + 子问题分解) |
| **编排** | LangGraph StateGraph (4 节点 + 条件边) |
| **Web** | FastAPI + SSE 真流式 (sse-starlette) |
| **前端** | Gradio ChatInterface |
| **会话** | SQLite 持久化 |
