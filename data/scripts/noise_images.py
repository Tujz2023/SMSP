import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import random

origin_images_dir = "output_images/5.png"
output_path = "output_images/5_noise.png"
noise_type = "vertical_gratings"  # Options: vertical_gratings, gaussian_noise, halftone_dots, labyrinth_noise, microtext_noise

def generate_vertical_gratings_noise(
    input_path,
    output_path,
    line_interval=20,
    line_width_bg=4,
    line_width_char=3,
    background=False
):
    if background:
        img = np.ones((1000, 1000), dtype=np.uint8) * 255
    else:
        if not os.path.exists(input_path):
            print(f"Error: File {input_path} does not exist")
            return None

        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Unable to read image {input_path}")
            return None

    _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    height, width = mask.shape
    
    canvas = np.ones((height, width), dtype=np.uint8) * 255
    
    for center_x in range(0, width, line_interval):
        col_data = mask[:, center_x]
        
        bg_rows = np.where(col_data > 127)[0]
        char_rows = np.where(col_data <= 127)[0]
        
        if len(bg_rows) > 0:
            x1 = max(0, center_x - line_width_bg)
            x2 = min(width, center_x)
            if x2 > x1:
                canvas[bg_rows[:, None], np.arange(x1, x2)] = 0
        
        if len(char_rows) > 0:
            x1 = max(0, center_x - line_width_char)
            x2 = min(width, center_x)
            if x2 > x1:
                canvas[char_rows[:, None], np.arange(x1, x2)] = 0

    if background:
        pil_image = Image.fromarray(canvas).convert("RGB")
        return pil_image

    cv2.imwrite(output_path, canvas)
    return None

def generate_gaussian_noise_image(
    input_path, 
    output_path, 
    char_gray_level=120,
    bg_gray_level=130,
    noise_std=120,
    background=False
):
    if background:
        img = np.ones((1000, 1000), dtype=np.uint8) * 255
    else:
        if not os.path.exists(input_path):
            print(f"Error: File {input_path} does not exist")
            return None
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Unable to read image {input_path}")
            return None
        
    mask_char = img < 127
    
    height, width = img.shape
    
    base_image = np.full((height, width), bg_gray_level, dtype=np.float32)
    
    base_image[mask_char] = char_gray_level
    
    noise = np.random.normal(0, noise_std, (height, width))
    
    final_image = base_image + noise
    
    final_image = np.clip(final_image, 0, 255)
    final_image = final_image.astype(np.uint8)

    if background:
        pil_image = Image.fromarray(final_image).convert("RGB")
        return pil_image
    
    cv2.imwrite(output_path, final_image)
    return None

def generate_halftone_dots_image(
    input_path, 
    output_path, 
    grid_size=20,
    bg_radius=3,
    fg_radius=2,
    position_jitter=7,
    shape_jitter=0,
    background=False
):
    if background:
        width, height = 1000, 1000
        img_array = np.full((height, width), 255, dtype=np.uint8)
    else:
        if not os.path.exists(input_path):
            print(f"Error: File {input_path} does not exist")
            return None

        original_img = Image.open(input_path).convert("L")
        width, height = original_img.size
        
        img_array = np.array(original_img)
    
    output_img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(output_img)
    
    for y in range(0, height, grid_size):
        for x in range(0, width, grid_size):
            y_end = min(y + grid_size, height)
            x_end = min(x + grid_size, width)
            
            patch = img_array[y:y_end, x:x_end]
            
            if patch.size == 0: continue
            mean_brightness = np.mean(patch)
            
            is_character = mean_brightness < 200
            
            base_radius = fg_radius if is_character else bg_radius
            
            if shape_jitter > 0:
                delta = int(base_radius * shape_jitter)
                r_noise = random.randint(-delta, delta)
                current_radius = max(1, base_radius + r_noise)
            else:
                current_radius = base_radius

            center_x = x + grid_size // 2
            center_y = y + grid_size // 2
            
            if position_jitter > 0:
                center_x += random.randint(-position_jitter, position_jitter)
                center_y += random.randint(-position_jitter, position_jitter)
            
            left = center_x - current_radius
            top = center_y - current_radius
            right = center_x + current_radius
            bottom = center_y + current_radius
            
            draw.ellipse((left, top, right, bottom), fill="black")

    if background:
        return output_img
    
    output_img.save(output_path)
    return None

def generate_labyrinth_noise_image(
    input_path,
    output_path,
    noise_scale=3,
    bias_strength=4,
    background=False
):
    if background:
        mask = np.ones((1000, 1000), dtype=np.uint8) * 255
    else:
        if not os.path.exists(input_path):
            print(f"Error: File {input_path} does not exist")
            return None
        mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    h, w = mask.shape
    
    mask_normalized = 1.0 - (mask / 255.0)
    raw_noise = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    
    k_size = (noise_scale * 4) + 1
    smooth_noise = cv2.GaussianBlur(raw_noise, (k_size, k_size), 0)
    smooth_noise = smooth_noise.astype(np.float32)
    signal_map = smooth_noise - (mask_normalized * bias_strength)
    median_val = np.median(smooth_noise)
    
    output_img = np.where(signal_map > median_val, 255, 0).astype(np.uint8)

    if background:
        pil_image = Image.fromarray(output_img).convert("RGB")
        return pil_image
    cv2.imwrite(output_path, output_img)
    return None

def generate_microtext_noise_image(
    input_path, 
    output_path, 
    font_size=14,
    density_jitter=7,
    background=False
):
    if background:
        w, h = 1000, 1000
        img_array = np.full((h, w), 255, dtype=np.uint8)
    else:
        if not os.path.exists(input_path):
            print(f"Error: File {input_path} does not exist")
            return None
        original_img = Image.open(input_path).convert("L")
        w, h = original_img.size
        img_array = np.array(original_img)
    
    output_img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(output_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    chars_bg = ['$', '%', '#']
    chars_fg = ['@', '&']
    
    step = int(font_size * 1.4) 
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            if y < h and x < w:
                is_char = img_array[y, x] < 128
            else:
                continue
            
            if is_char:
                char_to_draw = random.choice(chars_fg)
                text_color = (0, 0, 0)
            else:
                char_to_draw = random.choice(chars_bg)
                text_color = (0, 0, 0)

            offset_x = random.randint(-density_jitter, density_jitter) // 2
            offset_y = random.randint(-density_jitter, density_jitter) // 2
            
            draw_x = x + offset_x
            draw_y = y + offset_y
            
            draw.text((draw_x, draw_y), char_to_draw, font=font, fill=text_color)

    if background:
        return output_img
    
    output_img.save(output_path)
    return None

def main():

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if noise_type == 'vertical_gratings':
        generate_vertical_gratings_noise(input_path=origin_images_dir, output_path=output_path)
    elif noise_type == 'gaussian_noise':
        generate_gaussian_noise_image(input_path=origin_images_dir, output_path=output_path)
    elif noise_type == 'halftone_dots':
        generate_halftone_dots_image(input_path=origin_images_dir, output_path=output_path)
    elif noise_type == 'labyrinth_noise':
        generate_labyrinth_noise_image(input_path=origin_images_dir, output_path=output_path)
    elif noise_type == 'microtext_noise':
        generate_microtext_noise_image(input_path=origin_images_dir, output_path=output_path)
    else:
        print(f"Unsupported noise type: {noise_type}")
    

if __name__ == "__main__":
    main()