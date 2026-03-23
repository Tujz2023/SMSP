from openai import OpenAI

import json
import os
import time
from tqdm import tqdm
import multiprocessing
from functools import partial
import re

def split_and_parse_jsons(line):
    datas = []

    split_points = [match.start() for match in re.finditer(r"}{(?=\"data\")", line)]

    last_split = 0
    for split_point in split_points:
        json_str = line[last_split:split_point + 1]
        try:
            datas.append(json.loads(json_str))
        except json.JSONDecodeError:
            print("JSON解析错误:", json_str)

        last_split = split_point + 1

    if last_split < len(line):
        try:
            datas.append(json.loads(line[last_split:]))
        except json.JSONDecodeError:
            print("JSON解析错误:", line[last_split:])

    return datas

def get_answer(line, api_key, api_url, retry_count=15, save_path=None):
    payload, data = line[0], line[1]

    client = OpenAI(api_key=api_key, base_url=api_url)

    try:
        time.sleep(1)
        response = client.chat.completions.create(
            **payload
        )
        if 'error' in response and response['error']['code'] == 'rate_limit_exceeded':
            raise Exception

        output = response.choices[0].message.content

        if payload['model'] == "google/gemini-2.5-pro":
            if output == "":
                output = f"reasoning: {response.choices[0].message.reasoning}"

        data = {
            'data': data,
            'response': output
        }

    except Exception as e:
        print("发生了异常:", str(e))
        if retry_count > 0:
            print("等待3秒后进行重试...")
            time.sleep(3)
            return get_answer(
                line,
                api_key=api_key,
                api_url=api_url,
                retry_count=retry_count - 1,
                save_path=save_path
            )
        else:
            print("重试次数已达上限，无法获取答案。")
            return None
    
    with open(save_path,'a',encoding='utf-8') as f:
        print(json.dumps(data, ensure_ascii=False), file=f)
    return data

def generate(model_name, api_key, api_url, messages, data, output_path, max_tokens=8192, top_p=0.1, temperature=0., limit=None, system_prompt=None):

    if limit is not None:
        data = data[:limit]
        messages = messages[:limit]
    
    payloads = []
    
    if not os.path.exists(output_path):
        dir_path = os.path.dirname(output_path)
        os.makedirs(dir_path, exist_ok=True)
    
    if os.path.exists(output_path):
        gen_ids = set()
        with open(output_path) as f:
            for x, line in enumerate(f):
                if not line.strip():
                    continue

                for item in split_and_parse_jsons(line):
                    gen_ids.add(item['data']['id'])

        new_data = []
        
        for d in data:
            if d['id'] not in gen_ids:
                new_data.append(d)
                print(d["id"])

        print(f'total: {len(data)} samples, finished: {len(gen_ids)} samples, to be finished: {len(new_data)} samples')

        data = new_data
    
    print(f'total samples:{len(data)}')
    for i in range(len(data)):
        if model_name == "gpt-5.2" or model_name == "claude-sonnet-4-5-20250929":
            payload = {
                "model": model_name,
                "max_tokens": max_tokens,
                "messages": messages[i]
            }
        elif model_name == "google/gemini-2.5-pro":
            payload = {
                "model": model_name,
                "max_tokens": max_tokens,
                "reasoning_effort": "low",
                "messages": messages[i]
            }
        else:
            payload = {
                "model": model_name,
                "top_p": top_p,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": messages[i]
            }
        payloads.append([payload, data[i]])
        
    with multiprocessing.Pool(12) as pool:
        func = partial(
            get_answer,
            api_key=api_key,
            api_url=api_url,
            save_path=output_path
        )
        result = list(tqdm(pool.imap(func, payloads), total=len(payloads)))

    result = [i for i in result if i!= None]
