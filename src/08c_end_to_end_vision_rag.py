import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import uuid
import gc

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download

print("==================================================")
print("第一阶段：初始化 RAG 知识库基础组件")
print("==================================================")
print("1. 正在加载 BGE 向量模型与 ChromaDB 数据库...")
# 为了防止 Windows 底层 C++ 线程池崩溃，我们优先加载轻量级的本地向量库
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


print("\n==================================================")
print("第二阶段：多模态大模型出马，阅读并解析真实图片")
print("==================================================")
image_path = "test_chart.png" 

print("2. 正在加载 Qwen2-VL-2B 视觉大模型...")
model_id = snapshot_download('qwen/Qwen2-VL-2B-Instruct')
vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

print(f"3. 正在让模型阅读图片：{image_path}")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": "请详细描述这张图表中的所有数据信息，包括各个季度的收入情况。"},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
).to("cuda")

generated_ids = vlm_model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
vlm_parsed_result = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print("\n💡 VLM 自动解析出来的结果：")
print(vlm_parsed_result)

print("\n4. 🧹 解析完毕！正在自动卸载 VLM 模型，释放显存...")
del vlm_model
del processor
gc.collect()
torch.cuda.empty_cache()


print("\n==================================================")
print("第三阶段：无缝对接 RAG，写入结果并测试")
print("==================================================")
print("5. 正在把 VLM 的解析结论存入本地 ChromaDB 数据库...")
vector_db.add_texts(
    texts=["【自动化图表解析】" + vlm_parsed_result],
    metadatas=[{"source": image_path, "type": "image_description"}],
    ids=[str(uuid.uuid4())]
)
print("✅ 写入成功！")

print("6. 正在测试最终检索链路...")
question = "2023年第三季度的收入是多少？"
print(f"用户提问：{question}")

docs = vector_db.similarity_search(question, k=1)
print("\n🎉 知识库检索结果：")
print(docs[0].page_content)
print(f"(资料溯源: {docs[0].metadata.get('source')})")
print("\n🔥 完美闭环！接下来我们终于可以去启动炫酷网页了！")
