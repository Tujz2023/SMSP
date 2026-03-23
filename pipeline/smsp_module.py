from vllm import LLM, SamplingParams
from tqdm import tqdm
from load_inputs import *
from perception import perception_module
from api_utils import generate
import os
import json
from PIL import Image
from decimal import Decimal, getcontext
from api_utils import split_and_parse_jsons
from loguru import logger

def cal_params(t_0, t_k, k, precision=50):
    getcontext().prec = precision

    t_0 = Decimal(t_0)
    t_k = Decimal(t_k)
    k = Decimal(k)

    ratio = t_k / t_0
    seq = [t_0]

    for i in range(2, int(k)):
        exponent = (Decimal(i) - 1) / (k - 1)
        t_i = t_0 * (ratio ** exponent)
        seq.append(t_i)
    seq.append(t_k)

    seq = [float(t_i.quantize(Decimal('0.00001'))) for t_i in seq]

    return seq

def postprocess(path, outpath):
    datas = []
    with open(path, "r") as f:
        num = 0
        for line in f:
            # Skip empty lines
            if not line.strip():
                continue
            for item in split_and_parse_jsons(line):
                datas.append(item)
                
    result = []
    for example in datas:
        flag = True
        for res in result:
            if(res["id"] == example["data"]["id"]):
                flag = False
        if flag == False:
            continue

        data = example["data"]
        data["response"] = example["response"]
        result.append(data)
    
    result.sort(key=lambda x:x['id'])
    
    with open(outpath, 'w') as outf:
        json.dump(result, outf, ensure_ascii=False, indent=2)

def save_file(output_path, results):
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

class MultiScale_Perception:
    def __init__(self, model_name, percep_type, use_api, api_key, api_url, cache_path, model_path):
        self.model_name = model_name
        self.percep_type = percep_type
        self.use_api = use_api
        self.api_key = api_key
        self.api_url = api_url
        self.cache_path = cache_path
        self.model_path = model_path
        self.perception_module = perception_module

        if percep_type == 'smsp':
            with open('perception_params.json', 'r', encoding='utf-8') as f:
                params = json.load(f)
            self.percep_params = {
                "variant_num": params['variant_num'],
                "lambda": [],
                "s": []
            }
            self.percep_params['lambda'] = cal_params(params['lambda_0'], params['lambda_k'], params['variant_num'])
            self.percep_params['s'] = cal_params(params['s_0'], params['s_k'], params['variant_num'])

    def percep(self, data):
        cache_path = self.cache_path
        if os.path.exists(f'{cache_path}/metadata.json'):
            with open(f'{cache_path}/metadata.json', 'r', encoding='utf-8') as f:
                new_data = json.load(f)
            return new_data
        if cache_path != '':
            os.makedirs(cache_path, exist_ok=True)
        
        new_data = []
        i = 0
        for d in tqdm(data):
            image = d['image']

            if self.percep_type == 'filtered' or self.percep_type == 'blur_and_histogram':
                variant = self.perception_module(image, self.percep_type)
                if cache_path != '':
                    save_path = f'{cache_path}/{i:04d}.png'
                    variant.save(save_path)
                    new_data.append(save_path)
                else:
                    new_data.append(variant)
                i += 1

            else:
                for t in range(self.percep_params['variant_num']):
                    variant = self.perception_module(image, self.percep_type, self.percep_params['lambda'][t], self.percep_params['s'][t])
                    if cache_path != '':
                        save_path = f'{cache_path}/{i:04d}.png'
                        variant.save(save_path)
                        new_data.append(save_path)
                    else:
                        new_data.append(variant)
                    i += 1

        if cache_path != '':
            with open(f'{cache_path}/metadata.json', 'w', encoding='utf-8') as f:
                json.dump(new_data, f, ensure_ascii=False, indent=2)
        return new_data
    
    def generate(self, origin_data, output_path, percep_data=None):
        logger.info(f"Loading variants from cache, and processing inputs ...")
        if percep_data == None:
            if self.use_api == 'true':
                inputs, save_data = load_api_inputs(origin_data)
            else:
                inputs = load_qwen_inputs(origin_data, self.model_path)
        else:
            if self.cache_path != '':
                data_path = percep_data
                percep_data = []
                for path in tqdm(data_path):
                    percep_data.append(Image.open(path).convert("RGB"))
            if self.use_api == 'true':
                if self.percep_type != 'smsp':
                    inputs, save_data = load_api_inputs(origin_data)
                else:
                    inputs, save_data = load_api_inputs(origin_data, percep_data)
            else:
                if self.percep_type != 'smsp':
                    inputs = load_qwen_inputs(origin_data, self.model_path)
                else:
                    inputs = load_qwen_inputs(origin_data, self.model_path, percep_data)

        if self.use_api == 'true':
            self.api_generate(self.api_key, self.api_url, inputs, save_data, output_path)
            return
        else:
            self.local_generate(inputs, origin_data, output_path)

    def api_generate(self, api_key, api_url, messages, new_data, output_path):
        logger.info(f"Generating response ...")
        output_path = f"{output_path}l"
        generate(self.model_name, api_key, api_url, messages, new_data, output_path, max_tokens=8192, top_p=0.1, temperature=0.)
        postprocess(output_path, output_path[:-1])
        return

    def local_generate(self, inputs, origin_data, output_path):
        logger.info(f"Loading models ...")
        model = LLM(
            model=self.model_path,
            mm_encoder_tp_mode="data"
        )
        sampling_params = SamplingParams(
            max_tokens=1024,
            stop_token_ids=[],
        )

        logger.info(f"Generating response ...")
        outputs = model.generate(inputs, sampling_params=sampling_params)

        results = []
        i = 0
        for data in origin_data:
            data['response'] = outputs[i].outputs[0].text
            del data['image']
            results.append(data)
            i += 1
        
        save_file(output_path, results)