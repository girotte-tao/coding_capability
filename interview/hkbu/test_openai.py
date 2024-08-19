import os
from openai import OpenAI





client = OpenAI(
    api_key = tongyi_key,
    base_url= 'https://dashscope.aliyuncs.com/compatible-mode/v1'
)


def chat_gpt(client, prompt):
    prompt = prompt

    # completion = client.completions.create(
    #     model = "gpt-3.5-turbo-instruct",
    #     prompt = prompt,
    #     max_tokens = 7,
    #     temperature = 0
    # )
    
    completion = client.completions.create(
        model = "qwen-turbo",
        prompt = prompt,
        max_tokens = 7,
        temperature = 0
    )
    

    response = completion.choices[0].text
    print(response)
    
chat_gpt(client, "What is the capital of China?")




