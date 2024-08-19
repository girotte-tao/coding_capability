import openai


# 替换为你的 OpenAI API 密钥
openai.api_key = key

def test_openai_api():
    try:
        # 测试消息，可以根据需要修改
        prompt = "What is the capital of France?"
        
        # 发送请求给 OpenAI API
        response = openai.Completion.create(
            engine="text-davinci-003",  # 替换为你要使用的引擎
            prompt=prompt,
            max_tokens=50  # 设置生成的响应长度
        )
        
        # 获取并打印响应
        print("API Response:")
        print(response.choices[0].text.strip())

        # 返回响应文本以便进一步处理
        return response.choices[0].text.strip()
    
    except openai.error.OpenAIError as e:
        print(f"API request failed: {e}")
        return None

if __name__ == "__main__":
    result = test_openai_api()
    if result:
        print(f"Test passed, received response: {result}")
    else:
        print("Test failed.")
