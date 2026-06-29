import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

knowledge_doc = """
【公司研究报告：星辰科技】
星辰科技是一家专注于人工智能芯片研发的公司。
近期，公司发布了第三代 AI 推理芯片“星云-3”，算力相比上一代提升了 200%，功耗降低了 40%。
该芯片主要应用于自动驾驶和智能安防领域。
财务方面，公司2023年第三季度营业收入达到 15 亿元，同比增长 50%。
预计第四季度业绩将继续保持高速增长。
"""

print("1. ✂️ 正在进行文档切块 (Chunking)...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
chunks = text_splitter.split_text(knowledge_doc)

for i, chunk in enumerate(chunks):
    print(f"  [片段 {i}]: {chunk}")

print("\n2. 🧠 加载 Embedding 模型...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

print("\n3. 🗄️ 将文字片段存入 Chroma 向量数据库...")
# ChromaDB 对 Windows 非常友好，是最流行的本地向量数据库之一
vector_db = Chroma.from_texts(chunks, embeddings)

print("\n4. 🔍 开始魔法检索！")
question = "星辰科技第三季度的收入是多少？"
print(f"用户提问: {question}")

docs = vector_db.similarity_search(question, k=2)

print("\n✅ 检索到的最相关内容：")
for i, doc in enumerate(docs):
    print(f" -> 匹配结果 {i+1}: {doc.page_content}")
