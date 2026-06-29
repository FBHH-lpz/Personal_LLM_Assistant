import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelscope import snapshot_download

# ================= 1. 准备 RAG 数据库 =================
print("1. 🗄️ 正在建立 ChromaDB 知识库...")
knowledge_doc = """
【公司研究报告：星辰科技】
星辰科技是一家专注于人工智能芯片研发的公司。
近期，公司发布了第三代 AI 推理芯片“星云-3”，算力相比上一代提升了 200%，功耗降低了 40%。
该芯片主要应用于自动驾驶和智能安防领域。
财务方面，公司2023年第三季度营业收入达到 15 亿元，同比增长 50%。
预计第四季度业绩将继续保持高速增长。
"""
# 切块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
chunks = text_splitter.split_text(knowledge_doc)

# 转换向量并存入数据库
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cuda'}
)
vector_db = Chroma.from_texts(chunks, embeddings)


# ================= 2. 准备 Qwen 大模型 =================
print("\n2. 🤖 正在加载 Qwen2.5-0.5B 大脑到你的 RTX 4060 显卡上...")
model_id = snapshot_download('qwen/Qwen2.5-0.5B-Instruct')
tokenizer = AutoTokenizer.from_pretrained(model_id)
# 这里我们用 .to("cuda") 让大模型在你的显卡上飞速跑起来！
model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")


# ================= 3. RAG 核心：外挂大脑工作流 =================
question = "星辰科技第三季度的收入是多少？算力提升了多少？"
print(f"\n🙋 你的问题: {question}")

# 【第一步：检索】去知识库里搜索相关的文本片段
print("\n🔍 第一步：知识库检索中...")
docs = vector_db.similarity_search(question, k=2)

# 把搜索到的两个片段拼接到一起，作为大模型的“参考资料”
retrieved_context = "\n".join([doc.page_content for doc in docs])
print(f"✅ 找到的参考资料如下:\n{retrieved_context}\n")

# 【第二步：生成】把“参考资料”和“用户问题”一起喂给大模型
print("✍️ 第二步：大模型正在阅读资料并组织回答...")
messages = [
    {"role": "system", "content": "你是一个专业的分析师。请严格根据用户提供的[参考资料]来回答问题。如果资料里没有写，就说不知道。不要编造内容。回答要简明扼要。"},
    {"role": "user", "content": f"[参考资料]:\n{retrieved_context}\n\n请根据上述资料回答我的问题: {question}"}
]

# 将对话模板转换为模型能看懂的输入，并发送到显卡
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

# 让显卡开始推理生成回答
generated_ids = model.generate(**model_inputs, max_new_tokens=200)

# 过滤掉我们输入给模型的 prompt 部分，只保留它新生成的回答内容
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n🎉 大模型的最终回答：")
print("====================================")
print(response)
print("====================================")
