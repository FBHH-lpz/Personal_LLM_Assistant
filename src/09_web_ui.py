import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import warnings
warnings.filterwarnings('ignore') # 忽略一些无聊的终端警告

import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

print("⏳ 正在加载向量知识库与外挂大脑，这需要十几秒钟...")

# 1. 挂载记忆数据库 (RAG)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 2. 挂载语言大模型 (并且插上我们在阶段三训练好的 U盘)
model_id = "qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, "lora_model/qwen_lora_assistant")

print("✅ 大脑加载完毕！正在启动 Web 网页...")

# 3. 核心聊天逻辑
def _extract_text(obj):
    # 暴力提取纯文本，应对 Gradio 6 各种奇葩的多模态列表结构
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, (list, tuple)):
        return str(obj[0]) if len(obj) > 0 else ""
    elif isinstance(obj, dict):
        return obj.get("text", str(obj))
    return str(obj)

def chat_with_assistant(message, history):
    safe_message = _extract_text(message)
    
    # 优化点1：让检索支持多轮语境
    search_query = safe_message
    if len(history) > 0:
        last_item = history[-1]
        last_msg = last_item["content"] if isinstance(last_item, dict) else last_item[0]
        search_query = _extract_text(last_msg) + " " + safe_message

    # 第一步：根据用户输入，在知识库里搜索最相关的 2 条资料
    docs = vector_db.similarity_search(search_query, k=2)
    context = "\n".join([doc.page_content for doc in docs])
    
    # 第二步：把检索到的资料塞给系统 Prompt
    system_prompt = f"你是一个专业的个人AI助理。请参考以下资料回答用户的问题：\n[参考资料]：\n{context}\n\n如果资料中没有提到，请用你自己的知识回答。"
    
    # 构造上下文记忆
    messages = [{"role": "system", "content": system_prompt}]
    for item in history:
        if isinstance(item, dict):
            messages.append({"role": item["role"], "content": _extract_text(item["content"])})
        else:
            messages.append({"role": "user", "content": _extract_text(item[0])})
            messages.append({"role": "assistant", "content": _extract_text(item[1])})
            
    messages.append({"role": "user", "content": safe_message})
    
    # 第三步：生成回答
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    generated_ids = model.generate(**model_inputs, max_new_tokens=256)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

# 4. 构建漂亮的交互式网页
demo = gr.ChatInterface(
    fn=chat_with_assistant,
    title="🤖 我的专属大模型助理 (RAG + LoRA 版)",
    description="内核：Qwen2.5-0.5B | 知识库：ChromaDB | 微调：LoRA。\n快来向它提问吧！（比如问它：星辰科技第三季度的收入是多少？）"
)

if __name__ == "__main__":
    # 启动网页并在本地 7860 端口提供服务
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
