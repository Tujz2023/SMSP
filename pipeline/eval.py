import json
from gpt_eval import gpt_eval

input_file = "results/output.json"
already_eval = False
output_file = "results/output_eval.json"
result_file = "results/output_result.json"
api_key="your_api_key_here"         # change to your API key
api_url="https://your_api_url.com"  # change to your API URL
limit = None
analysis = True

results = {
    "large": {
        "origin": [0, 0],
        "noise": [0, 0],
        "semantic": [0, 0]
    },
    "medium": {
        "origin": [0, 0],
        "noise": [0, 0],
        "semantic": [0, 0]
    },
    "small": {
        "origin": [0, 0],
        "noise": [0, 0],
        "semantic": [0, 0]
    }
}

def main():
    input_path = input_file
    if not already_eval:
        gpt_eval(input_file, output_file, api_key, api_url, limit=limit, analysis=analysis)
        input_path = output_file
    
    with open(input_path, 'r', encoding='utf-8') as f:
        ori_results = json.load(f)

    for item in ori_results:

        size = item['size']
        if item['noise_type'] == 'vertical_line' or item['noise_type'] == 'gaussian_noise' or item['noise_type'] == 'halftone_noise' or item['noise_type'] == 'labyrinth_noise' or item['noise_type'] == 'microtext_noise':
            results[size]['noise'][1] += 1
            if item['is_correct']:
                results[size]['noise'][0] += 1
        elif item['noise_type'] == 'chinese_architecture' or item['noise_type'] == 'cyberpunk_city' or item['noise_type'] == 'winter_valley':
            results[size]['semantic'][1] += 1
            if item['is_correct']:
                results[size]['semantic'][0] += 1
        else:
            results[size]['origin'][1] += 1
            if item['is_correct']:
                results[size]['origin'][0] += 1

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()