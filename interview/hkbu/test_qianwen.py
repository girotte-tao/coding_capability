from openai import OpenAI
import os

def get_response():
    tongyi_key = 'sk-4efe75b63309400ab930120a86024856'

    client = OpenAI(
        api_key=tongyi_key, # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务的base_url
    )
    completion = client.chat.completions.create(
        model="qwen-turbo",
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': '你是谁？'}],
        
        temperature=0.8,
        top_p=0.8
        )
    print(completion.model_dump_json())
    

def get_prommpt_response():
    tongyi_key = ''

    client = OpenAI(
        api_key=tongyi_key, # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务的base_url
    )
    completion = client.chat.completions.create(
        model="qwen-turbo",
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': '你是谁？'}],
        
        temperature=0.8,
        top_p=0.8
        )
    print(completion.model_dump_json())
    


if __name__ == '__main__':
    get_response()