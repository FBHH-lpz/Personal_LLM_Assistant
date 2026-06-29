import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import json
import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download

print("==================================================")
print(" 第一部分：手写本地的外挂工具 (Tools)")
print("==================================================")

# 1. 定义真实的 Python 工具函数
def get_current_time(**kwargs):
    """返回当前时间"""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"当前的真实时间是: {now}"

def calculate_math(expression, **kwargs):
    """安全地计算数学公式"""
    try:
        # 出于安全考虑，真实环境会用 AST 安全评估，这里为了教学直接使用 eval
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {str(e)}"

# 将工具映射到字典，方便后续调用
TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "calculate_math": calculate_math
}

# 2. 给大模型发送的“工具说明书” (JSON Schema 格式)
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的系统时间和日期，当你需要回答关于时间的问题时调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_math",
            "description": "计算数学表达式的结果。当你需要进行加减乘除等数学运算时，必须调用此工具，不要自己瞎算。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的 Python 数学表达式，例如 '123 * 456' 或者 '999 + 888'"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

print("✅ 工具准备完毕。")

print("\n==================================================")
print(" 第二部分：加载大模型 (Qwen2.5 原味版)")
print("==================================================")
model_dir = snapshot_download('qwen/Qwen2.5-0.5B-Instruct')
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir, torch_dtype=torch.float16, device_map="auto"
)


print("\n==================================================")
print(" 第三部分：开启 Agent 代理循环")
print("==================================================")

user_question = "你好！请告诉我现在的准确时间是几点？还有，请帮我算一下 999 乘以 888 减去 1234 等于多少？"
print(f"🙋‍♂️ 用户提问: {user_question}")

# 将工具库塞入对话模板的系统 prompt 中
messages = [{"role": "user", "content": user_question}]

# 【第一次思考】：让模型判断是否需要调用工具
print("\n🧠 模型第一次思考中 (正在判断是否需要使用工具)...")
text = tokenizer.apply_chat_template(messages, tools=tools_schema, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
response_text = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

print(f"\n💬 模型的底层原始输出截获:\n{response_text}")


# 4. 解析工具调用并执行本地代码
if "<tool_call>" in response_text:
    print("\n🚨 触发拦截！模型发现自己无法回答，主动请求调用外部 Python 工具！")
    
    # 把它刚才说的这段带 <tool_call> 的长文本加入记忆中
    messages.append({"role": "assistant", "content": response_text})
    
    # 暴力解析 XML 标签里的 JSON (仅作教学，生产环境可用正则)
    calls = response_text.split("<tool_call>")
    
    for call in calls[1:]:
        tool_json_str = call.split("</tool_call>")[0].strip()
        tool_call_data = json.loads(tool_json_str)
        
        func_name = tool_call_data["name"]
        arguments = tool_call_data.get("arguments", {})
        
        print(f"🔧 [执行动作]: 正在本地运行代码 ---> {func_name}({arguments})")
        
        # 真正去执行我们在最上方定义的 Python 函数！
        result = TOOL_FUNCTIONS[func_name](**arguments)
        print(f"✅ [得到结果]: {result}")
        
        # 5. 极其关键的一步：将结果以 role="tool" 的身份“喂回”给大模型
        messages.append({
            "role": "tool", 
            "name": func_name, 
            "content": str(result)
        })

    # 【第二次思考】：模型结合拿到的工具结果，生成最终人类语言
    print("\n🧠 数据回传成功，模型第二次思考中 (融合数据进行总结)...")
    text2 = tokenizer.apply_chat_template(messages, tools=tools_schema, tokenize=False, add_generation_prompt=True)
    inputs2 = tokenizer([text2], return_tensors="pt").to("cuda")
    
    gen_ids2 = model.generate(**inputs2, max_new_tokens=256)
    gen_trimmed2 = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs2.input_ids, gen_ids2)]
    final_answer = tokenizer.batch_decode(gen_trimmed2, skip_special_tokens=True)[0]
    
    print("\n🎉 模型最终完美回复:")
    print(final_answer)

else:
    print("模型没有调用工具，直接给出了回答。")
