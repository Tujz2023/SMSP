dataset_path=data/
dataset_repo=Tujz/IlluChar
output_path=results/output.json
cache_path=cache

model_name=Qwen3-VL-4B-Instruct

use_api=true # [true, false]
percep_type=smsp # [smsp, blur_and_histogram, filtered_image, cot, vanilla]

model_path=/dir/to/your/model       # change to your local model path

api_key="your_api_key_here"         # change to your API key
api_url="https://your_api_url.com"  # change to your API URL

if [ "$use_api" = "true" ] ; then
    echo "Using API for inference"
    CUDA_VISIBLE_DEVICES=0 python smsp.py --use_api $use_api --api_key $api_key --api_url $api_url --dataset_repo $dataset_repo --dataset_path $dataset_path --output_path $output_path --cache_path $cache_path --model_name $model_name --percep_type $percep_type
    exit 0
fi

echo "Using local model for inference"
CUDA_VISIBLE_DEVICES=0 python smsp.py --use_api $use_api --model_path $model_path --dataset_repo $dataset_repo --dataset_path $dataset_path --output_path $output_path --cache_path $cache_path --model_name $model_name --percep_type $percep_type