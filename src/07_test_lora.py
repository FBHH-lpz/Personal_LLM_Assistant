import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from modelscope import snapshot_download

print("1. 🤖 正在加载 '原汁原味' 的 Qwen2.5-0.5B 大脑...")
model_id = snapshot_download('qwen/Qwen2.5-0.5B-Instruct')
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)

print("2. 💾 正在把我们刚刚训练好的 '专属记忆 U盘' (LoRA 侧脑) 插上去...")
# 这就是 Peft 的魔法，直接把 U盘 插到原版模型上，不需要改动原版代码！
model = PeftModel.from_pretrained(base_model, "lora_model/qwen_lora_assistant")

print("\n3. 🔍 准备进行终极测试！")
# 这是我们在 04_full_rag.py 里问过的问题，当时它把“算力提升了多少”给漏答了
question = "星辰科技第三季度的收入是多少？算力提升了多少？"

# 这也是我们在 04_full_rag.py 里通过 ChromaDB 检索出的参考资料
retrieved_context = """
财务方面，公司2023年第三季度营业收入达到 15 亿元，同比增长 50%。
近期，公司发布了第三代 AI 推理芯片“星云-3”，算力相比上一代提升了 200%，功耗降低了
"""

messages = [
    {"role": "system", "content": "你是一个专业的分析师。请严格根据用户提供的[参考资料]来回答问题。如果资料里没有写，就说不知道。不要编造内容。回答要简明扼要。"},
    {"role": "user", "content": f"[参考资料]:\n{retrieved_context}\n\n请根据上述资料回答我的问题: {question}"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

print("✍️ 挂载了新记忆的大模型正在组织回答...")
generated_ids = model.generate(**model_inputs, max_new_tokens=100)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n🎉 经过微调后的最终回答：")
print("====================================")
print(response)
print("====================================")
