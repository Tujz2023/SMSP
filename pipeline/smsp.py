import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset

from loguru import logger
import argparse
from tqdm import tqdm
from prompt import PROMPT, MULTI_SCALE_PROMPT, COT_PROMPT
from smsp_module import MultiScale_Perception
import json
from PIL import Image

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate responses using a pretrained model.")
    parser.add_argument('--use_api', type=str, required=True, default='false', help='Whether to use API for generation')
    parser.add_argument('--model_path', type=str, required=False, default='', help='Path to the model')
    parser.add_argument('--dataset_repo', type=str, required=True, help='Repository name of the dataset')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the input data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output file')
    parser.add_argument('--cache_path', type=str, required=False, default='', help='Path to save cache')
    parser.add_argument('--api_key', type=str, required=False, default='', help='API key if using API')
    parser.add_argument('--api_url', type=str, required=False, default='', help='API URL if using API')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--percep_type', type=str, required=True, default='smsp', help='Method of perception, including [smsp, blur_and_histogram, filtered_image, cot, vanilla]')
    parser.add_argument('--limit', type=int, required=False, default=-1, help='Limit of the input data')
    return parser.parse_args()

def load_data(dataset_repo, dataset_path, limit):
    if limit == -1:
        dataset = load_dataset(dataset_repo, split="train", cache_dir=dataset_path)
    else:
        dataset = load_dataset(dataset_repo, split=f"train[:{limit}]", cache_dir=dataset_path)
    return dataset

    # with open(f"{dataset_path}/metadata.jsonl", 'r', encoding='utf-8') as f:
    #     datas = [json.loads(line) for line in f.readlines()]
    # datas = datas[:limit] if limit != -1 else datas
    # dataset = []
    # for data in tqdm(datas):
    #     temp_data = data
    #     temp_data['image'] = Image.open(f"{dataset_path}/{data['file_name']}")
    #     dataset.append(temp_data)
    # return dataset

def processing_prompt(datas, percep_type):
    new_data = []
    i = 0
    for data in tqdm(datas):

        data_temp = data.copy()
        c = data['character']
        if len(c) == 1:
            c = c[0]
            if c >= '0' and c <= '9':
                temp_type = "number"
            elif c >= 'A' and c <= 'Z':
                temp_type = "uppercase letter"
            elif c >= 'a' and c <= 'z':
                temp_type = "lowercase letter"
            else:
                temp_type = "Chinese character"
        else:
            if all((char >= '0' and char <= '9') for char in c):
                temp_type = "string of numbers"
            elif all((char >= 'a' and char <= 'z') or (char >= 'A' and char <= 'Z') for char in c):
                temp_type = "word"
            else:
                temp_type = "string of Chinese characters"
        if percep_type == 'vanilla' or percep_type == 'blur_and_histogram' or percep_type == 'filtered':
            if data['noise_type'] != 'null':
                temp_type = temp_type + " hidden"
            prompt = PROMPT.replace("[hidden_type]", temp_type)
        elif percep_type == 'cot':
            if data['noise_type'] != 'null':
                prompt = COT_PROMPT.replace("[hidden_type]", temp_type)
            else:
                prompt = PROMPT.replace("[hidden_type]", temp_type)
        elif percep_type == 'smsp':
            prompt = MULTI_SCALE_PROMPT.replace("[hidden_type]", temp_type)
        else:
            print("Invalid perception type.")
            return None

        data_temp['prompt'] = prompt
        data_temp['id'] = i

        new_data.append(data_temp)
        i += 1

    return new_data

def main():
    args = parse_arguments()

    logger.info(f"Loading data from {args.dataset_path}, and processing prompts ...")
    dataset = load_data(args.dataset_repo, args.dataset_path, args.limit)
    new_data = processing_prompt(dataset, args.percep_type)
    print("length of input data: ", len(new_data))

    smsp_model = MultiScale_Perception(args.model_name, args.percep_type, args.use_api, args.api_key, args.api_url, args.cache_path, args.model_path)

    if args.percep_type != 'vanilla' and args.percep_type != 'cot':
        logger.info(f"Processing perceptually adjusted variants using SMSP ...")
        percep_data = smsp_model.percep(new_data)
        smsp_model.generate(new_data, args.output_path, percep_data)
    else:
        smsp_model.generate(new_data, args.output_path)

if __name__ == "__main__":
    main()