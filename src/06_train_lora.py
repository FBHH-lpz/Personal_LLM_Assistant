import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from modelscope import snapshot_download

print("1. 📚 加载我们在上一步生成的训练数据集...")
dataset = load_dataset("json", data_files="data/train.jsonl", split="train")

print("2. 🤖 加载我们要教的 '学生'：Qwen2.5-0.5B-Instruct...")
model_id = snapshot_download('qwen/Qwen2.5-0.5B-Instruct')
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)

def format_chat_template(example):
    example["text"] = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return example

dataset = dataset.map(format_chat_template)

print("\n3. 🧠 最核心的部分：安装 LoRA '外挂侧脑' ...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print("来看看外挂侧脑后，我们要训练多少参数：")
model.print_trainable_parameters()  

print("\n4. 🎓 配置老师 (Trainer) 并开始上课！")
# 在最新的 TRL 库中，训练参数配置统一使用 SFTConfig
training_args = SFTConfig(
    output_dir="./lora_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=1,
    num_train_epochs=15,
    optim="adamw_torch",
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)

print("🔥 开始训练！这将会调用你的 RTX 4060 满负荷运转几秒钟！")
trainer.train()

print("\n5. 💾 训练完成，正在把 '侧脑' 单独保存下来...")
trainer.save_model("lora_model/qwen_lora_assistant")
print("✅ 恭喜！你的专属微调侧脑已经保存在了 lora_model/qwen_lora_assistant 文件夹里！")
