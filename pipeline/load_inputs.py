from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from tqdm import tqdm
import base64
from io import BytesIO

def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }

def load_qwen_inputs(data, model_path, percep_data=None):
    processor = AutoProcessor.from_pretrained(model_path)

    messages = []
    if percep_data is not None:
        variants_num = len(percep_data) // len(data)
    i = 0
    for d in tqdm(data):
        if percep_data is None:
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": d['image']},
                        {"type": "text", "text": d['prompt']},
                    ],
                }
            ])
            i += 1
        else:
            contents = []
            contents.append({"type": "image", "image": d['image']})
            for j in range(variants_num):
                contents.append({"type": "image", "image": percep_data[i + j]})
            contents.append({"type": "text", "text": d['prompt']})
            messages.append([
                {
                    "role": "user",
                    "content": contents
                }
            ])
            i += variants_num

    inputs = [prepare_inputs_for_vllm(message, processor) for message in tqdm(messages)]
    return inputs

def image_to_base64(pil_image):
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    byte_data = buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode("utf-8")
    return "data:image/png;base64," + base64_str

def load_api_inputs(data, percep_data=None):
    messages = []
    new_data = []
    if percep_data is not None:
        variants_num = len(percep_data) // len(data)

    i = 0
    for d in tqdm(data):
        if percep_data is None:
            messages.append([{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_base64(d['image'])}},
                    {"type": "text", "text": d['prompt']}
                ]
            }])
            i += 1
        else:
            contents = []
            contents.append({"type": "image_url", "image_url": {"url": image_to_base64(d['image'])}})
            for j in range(variants_num):
                contents.append({"type": "image_url", "image_url": {"url": image_to_base64(percep_data[i + j])}})
            contents.append({"type": "text", "text": d['prompt']})
            messages.append([
                {
                    "role": "user",
                    "content": contents
                }
            ])
            i += variants_num

        new_data.append({
            "id": d['id'],
            "prompt": d['prompt'],
            "character": d['character'],
            "character_type": d['character_type'],
            "noise_type": d['noise_type'],
            "size": d['size'],
            "file_name": d['file_name']
        })
    return messages, new_data