import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download

# ================= 1. 准备一张测试图片 =================
import matplotlib.pyplot as plt

print("1. 🎨 正在用代码生成一张测试用的柱状图图片...")
plt.figure(figsize=(6, 4))
# 模拟一份各季度的财报收入图表
plt.bar(['Q1', 'Q2', 'Q3', 'Q4'], [10, 15, 20, 25], color=['blue', 'orange', 'green', 'red'])
plt.title("2023 Revenue (in millions)")
plt.savefig("test_chart.png")
print("✅ 生成成功！图片已保存在当前目录下的 test_chart.png，你可以打开看看它长什么样。")


# ================= 2. 加载多模态大模型 =================
print("\n2. 👁️ 正在下载并加载带有视觉神经的多模态大模型 Qwen2-VL-2B...")
print("（这个模型有 20 亿参数，大约需要下载 4GB，请耐心等待它被加载到 RTX 4060 上...）")
model_id = snapshot_download('qwen/Qwen2-VL-2B-Instruct')

# 这里我们指定用 torch.float16 半精度加载，完美适配 8GB 显存
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)
# Processor 相当于眼睛和嘴巴，负责把图片像素和文字变成大模型懂的向量
processor = AutoProcessor.from_pretrained(model_id)


# ================= 3. 开始看图说话 =================
print("\n3. 🖼️ 把刚才生成的图片喂给模型，并向它提问...")
messages = [
    {
        "role": "user",
        "content": [
            # 传入我们生成的图片
            {"type": "image", "image": "test_chart.png"},
            # 提出问题
            {"type": "text", "text": "请分析一下这张图表里展示了什么数据？Q3的收入是多少？哪一个季度的收入最高？"},
        ],
    }
]

# 把图文混合格式转化成模型输入
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

print("🤔 模型正在用它的 '火眼金睛' 观察图表并思考...")
generated_ids = model.generate(**inputs, max_new_tokens=128)

# 剥离掉问题的 Prompt，只保留大模型的回答
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print("\n🎉 模型的视觉识别最终结果：")
print("====================================")
print(output_text[0])
print("====================================")
