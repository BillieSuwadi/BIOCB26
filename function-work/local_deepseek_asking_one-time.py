import requests
import json

# 1. 配置 API 地址和模型参数
API_URL = "http://localhost:11434/api/generate"
headers = {"Content-Type": "application/json"}
data = {
    "model": "deepseek",  # 模型名，必须和 ollama list 输出的一致（如 deepseek:7b）
    "prompt": "用Python写一个简单的二分查找算法，带注释",  # 你的提问
    "stream": False,  # 关闭流式输出，直接返回完整结果
    "temperature": 0.7  # 随机性，0-1 之间，越小越精准
}

# 2. 发送请求并获取结果
try:
    response = requests.post(API_URL, headers=headers, data=json.dumps(data))
    response.raise_for_status()  # 捕获 HTTP 错误
    result = response.json()
    # 3. 提取并打印回答
    print("DeepSeek 回答：\n", result["response"])
except Exception as e:
    print("调用失败：", str(e))