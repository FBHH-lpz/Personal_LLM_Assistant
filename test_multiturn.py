import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.09_web_ui import chat_with_assistant
    print("--- 第一次对话 ---")
    ans1 = chat_with_assistant("星辰科技的收入是多少", [])
    print("回答:", ans1)
    
    print("\n--- 第二次多轮对话 ---")
    # 模拟 Gradio 的多轮对话传入的数据结构
    ans2 = chat_with_assistant("那算力提升了多少", [["星辰科技的收入是多少", ans1]])
    print("回答:", ans2)
except Exception as e:
    import traceback
    traceback.print_exc()
