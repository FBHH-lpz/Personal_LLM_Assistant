# Personal LLM Assistant

A hands-on learning project that progressively builds a personal AI assistant — from a bare LLM pipeline all the way to a full agent with tool calling, RAG, LoRA fine-tuning, and a Gradio web UI.

Built around **Qwen2.5-0.5B-Instruct**, designed to run locally on a consumer GPU (tested on NVIDIA RTX 4060).

## Learning Path

Each script builds on the last, introducing one new concept at a time:

| Script | What it teaches |
|---|---|
| `01_hello_llm.py` | Download and run your first LLM pipeline via ModelScope |
| `02_embeddings.py` | Text embeddings and semantic similarity |
| `03_rag_chroma.py` | Retrieval-Augmented Generation with ChromaDB |
| `04_full_rag.py` | End-to-end RAG: ingestion → retrieval → generation |
| `05_prepare_dataset.py` | Format conversation data for fine-tuning |
| `06_train_lora.py` | LoRA fine-tuning on a custom persona (QLoRA + 4-bit) |
| `07_test_lora.py` | Load and test your fine-tuned adapter |
| `08_vision_llm.py` | Vision-language model for chart/image understanding |
| `08b_multimodal_rag.py` | Multimodal RAG: images + text in the knowledge base |
| `08c_end_to_end_vision_rag.py` | End-to-end vision RAG pipeline |
| `09_web_ui.py` | Gradio chat interface with RAG + LoRA integration |
| `10_agent_tool_calling.py` | LLM agent with native tool calling (time, calculator) |

## Tech Stack

- **Model**: Qwen2.5-0.5B-Instruct (via ModelScope)
- **Inference**: Hugging Face Transformers + PyTorch
- **Fine-tuning**: PEFT (LoRA/QLoRA), BitsAndBytes 4-bit quantization
- **RAG**: LangChain + ChromaDB + BGE embeddings
- **Vision**: Qwen2.5-VL
- **Web UI**: Gradio
- **Notebooks**: Jupyter (`notebooks/`)

## Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

## Hardware Requirements

- NVIDIA GPU with at least 6 GB VRAM (8 GB recommended)
- For vision scripts: additional VRAM headroom recommended

## Running

Start from the beginning and work your way up:

```bash
python src/01_hello_llm.py       # Your first LLM call
python src/09_web_ui.py          # Launch the Gradio web assistant
python src/10_agent_tool_calling.py  # Agent with tool calling
```

Open the walkthrough notebook for a guided tour:

```bash
jupyter notebook notebooks/walkthrough.ipynb
```
