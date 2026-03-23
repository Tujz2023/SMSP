import json
import os
import re
from prompt import EVAL_PROMPT
from api_utils import generate, split_and_parse_jsons

def is_correct(response):
    response = response.strip().lower()
    if 'incorrect' in response or 'no' in response or 'false' in response:
        return False
    elif 'correct' in response or 'true' in response or 'yes' in response:
        return True
    else:
        print(response)
        return False

def postprocess(path, outpath, result_data):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write('')

    datas = []
    with open(path, "r") as f:
        num = 0
        for line in f:
            # Skip empty lines
            if not line.strip():
                continue
            for item in split_and_parse_jsons(line):
                datas.append(item)
                
    for example in datas:
        flag = True
        for res in result_data:
            if(res["id"] == example["data"]["id"]):
                flag = False
        if flag == False:
            continue

        data = example["data"]
        data["eval_response"] = example["response"]
        data["is_correct"] = is_correct(example["response"])
        result_data.append(data)
    
    result_data.sort(key=lambda x:x['id'])
    
    with open(outpath, 'w') as outf:
        json.dump(result_data, outf, ensure_ascii=False, indent=2)

def gpt_eval(input_file, output_file, api_key, api_url, limit=None, analysis=False):
    save_path = f"{output_file}l"
    model = "gpt-4o"

    with open(input_file) as f:
        datas = json.load(f)

    messages = []
    new_data = []
    result_data = []
    for data in datas:
        without_sym  = re.sub(r'[^\w\u4e00-\u9fff]+', '', data['response'])
        if data['character'] not in without_sym:
            data['is_correct'] = False
            result_data.append(data)
            continue
        
        if data['size'] != 'large' and data['character'] in without_sym and data['character'] not in ['and', 'the', 'yes', 'you', 'for', 'not']:
            data['is_correct'] = True
            result_data.append(data)
            continue

        prompt = EVAL_PROMPT.replace('[GROUND_TRUTH]', data['character']).replace('[RESPONSE]', data['response'])
        messages.append([{
            "role": "user",
            "content": prompt
        }])
        new_data.append(data)

    print("total to be evaluated:", len(new_data))
    print("already evaluated:", len(result_data))
    # input()

    generate(model, api_key, api_url, messages, new_data, save_path, max_tokens=2048, top_p=0.1, temperature=0., limit=limit)
    postprocess(save_path, save_path[:-1], result_data)