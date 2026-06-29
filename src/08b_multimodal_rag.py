import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import uuid

print("1. 🧠 正在加载我们之前建好的 ChromaDB 向量知识库...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

print("\n2. 🔗 将 VLM (视觉模型) 解析出来的图表结果融入知识库...")
# 这是刚才多模态模型看图后得出的精炼结论
vlm_parsed_result = "【图表解析：2023年四个季度的收入柱状图】这张图表展示了2023年四个季度的收入（以百万美元为单位）。Q3的收入是20百万美元，是四个季度中最高的。"

# 把这段纯文本作为一个新的“文档片段”存入我们的向量数据库
vector_db.add_texts(
    texts=[vlm_parsed_result],
    metadatas=[{"source": "test_chart.png", "type": "image_description"}],
    ids=[str(uuid.uuid4())]
)
print("✅ 成功！图表信息已经化为文本向量，永久保存在了知识库中！")

print("\n3. 🔍 验证 RAG 检索链路...")
print("（注意：这里我们不需要再去加载那个 4GB 庞大的视觉大模型了，直接走普通文本检索即可！）")
question = "2023年第三季度(Q3)的收入是多少百万美元？哪一个季度收入最高？"
print(f"用户提问：{question}")

# 测试相似度检索
docs = vector_db.similarity_search(question, k=1)

print("\n🎉 ChromaDB 检索到的相关知识库片段：")
print("====================================")
print(docs[0].page_content)
print(f"(来源: {docs[0].metadata.get('source')})")
print("====================================")
print("\n💡 结论：以后遇到再复杂的图表，只要先让 VLM 看一遍转化为文字总结，再存入 RAG 数据库。以后普通的轻量级大模型就能瞬间秒答图表相关的问题！")
