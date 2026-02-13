import requests
import json

API_URL = "http://localhost:11434/api/generate"
data = {
    "model": "deepseek",
    "prompt": "解释一下数据库事务的ACID特性，用通俗的语言",
    "stream": True,  # 开启流式输出
    "temperature": 0.5
}

# 流式接收响应
try:
    response = requests.post(API_URL, json=data, stream=True)
    response.raise_for_status()
    print("DeepSeek 流式回答：\n", end="")
    # 逐行解析流式数据
    for line in response.iter_lines():
        if line:
            line_data = json.loads(line)
            if "response" in line_data:
                print(line_data["response"], end="", flush=True)  # 逐字打印
    print("\n")  # 最后换行
except Exception as e:
    print("调用失败：", str(e))