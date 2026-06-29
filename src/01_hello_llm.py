from transformers import pipeline
from modelscope import snapshot_download

print("正在使用阿里魔搭社区 (ModelScope) 下载并加载模型（国内直连速度非常快，请稍候）...")

# 1. 国内极速下载方案：使用 ModelScope
# 下载 Qwen2.5-0.5B-Instruct 权重到本地缓存目录
model_dir = snapshot_download('qwen/Qwen2.5-0.5B-Instruct')

# 2. 组装“魔法棒” (Pipeline)
# 我们将刚刚下载好的本地路径传递给模型，让它加载到你的第一块显卡 (RTX 4060) 上
generator = pipeline(
    "text-generation", 
    model=model_dir, 
    device=0 
)

print("\n🎉 模型加载完成！开始对话：")

# 2. 准备问题
# 大模型现在主流的对话格式都是这样的列表：包含角色 (role) 和内容 (content)
messages = [
    {"role": "user", "content": "你好！请用通俗易懂的一句话向我介绍一下什么是大语言模型（LLM）。"}
]

print("你问：", messages[0]["content"])
print("思考中...")

# 3. 让模型生成回答
# max_new_tokens 控制模型最多能吐出多少个字
output = generator(messages, max_new_tokens=100)

print("\n🤖 模型的回答：")
# 返回的结果是一个列表，我们提取里面最新生成的那段回复
print(output[0]['generated_text'][-1]['content'])
