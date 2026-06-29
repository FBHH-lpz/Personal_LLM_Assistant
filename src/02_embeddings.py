import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' 

from modelscope import snapshot_download
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from numpy.linalg import norm

print("正在通过 ModelScope 下载并加载 Embedding 模型...")
model_dir = snapshot_download('BAAI/bge-small-zh-v1.5')
print(f"模型路径: {model_dir}")

print("开始加载模型到 GPU (如果在这里瞬间闪退，99%是因为刚才下载模型文件时断网导致文件损坏了)...")
embeddings = HuggingFaceEmbeddings(
    model_name=model_dir,
    model_kwargs={'device': 'cuda'}, 
    encode_kwargs={'normalize_embeddings': True}
)
print("✅ 模型加载成功！")

text1 = "深度学习在计算机视觉中的目标检测取得了很大成功。"
text2 = "利用神经网络可以有效地识别图片里有哪些物体。"
text3 = "今天中午去公司楼下的食堂吃红烧排骨。"

print("\n🔄 开始将文本转化为向量 (Embedding)...")
vec1 = embeddings.embed_query(text1)
vec2 = embeddings.embed_query(text2)
vec3 = embeddings.embed_query(text3)

print(f"text1 转化后的向量长度为: {len(vec1)} 维的浮点数数组")
print(f"数组的前三个数字是: {vec1[:3]}...")

def cosine_similarity(A, B):
    return np.dot(A, B) / (norm(A) * norm(B))

sim_1_2 = cosine_similarity(vec1, vec2)
sim_1_3 = cosine_similarity(vec1, vec3)

print("\n📏 计算语义相似度：")
print(f"【句子1】和【句子2】的相似度: {sim_1_2:.4f}")
print(f"【句子1】和【句子3】的相似度: {sim_1_3:.4f}")
