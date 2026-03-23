from PIL import Image, ImageDraw, ImageFont
import os

input_text = "5"
output_path = "output_images/5.png"
char_size = 700
image_size = 1000
char_path = "C:\\Windows\\Fonts\\simhei.ttf" # default font library path for Windows

def single(single_text, save=True):
    render_char_size = char_size * 4
    try:
        char = ImageFont.truetype(char_path, size=render_char_size)
    except IOError:
        raise IOError(f"Cannot load character file: {char_path}. Please check if it is a valid character file.")

    bbox = char.getbbox(single_text)

    char_width = bbox[2] - bbox[0]
    char_height = bbox[3] - bbox[1]

    char_img = Image.new('L', (char_width, char_height), color=255)
    draw = ImageDraw.Draw(char_img)
    
    draw.text((-bbox[0], -bbox[1]), single_text, font=char, fill=0)

    if char_width > char_height:
        new_width = char_size
        new_height = int(char_height * (new_width / char_width))
    else:
        new_height = char_size
        new_width = int(char_width * (new_height / char_height))
    
    new_width = max(1, new_width)
    new_height = max(1, new_height)

    resized_char_img = char_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    background = Image.new('L', (image_size, image_size), color=255)

    bg_width, bg_height = background.size
    paste_x = (bg_width - new_width) // 2
    paste_y = (bg_height - new_height) // 2

    background.paste(resized_char_img, (paste_x, paste_y))
    
    output_dir = os.path.dirname(output_path)
    
    if save:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        background.save(output_path)

    return background

def comb(comb_text):

    processed_imgs = []
    total_cropped_width = 0

    n = len(comb_text)
    
    for i, single_text in enumerate(comb_text):
        single_img = single(single_text, save=False).convert("RGB")

        if n == 1:
            crop_box = (100, 0, 900, 1000)
        elif i == 0:
            crop_box = (0, 0, 900, 1000)
        elif i == n - 1:
            crop_box = (100, 0, 1000, 1000)
        else:
            crop_box = (100, 0, 900, 1000)

        cropped_img = single_img.crop(crop_box)
        processed_imgs.append(cropped_img)
        total_cropped_width += cropped_img.width

    if not processed_imgs:
        return

    scale_ratio = image_size / total_cropped_width
    
    new_strip_height = int(image_size * scale_ratio)

    bg_img = Image.new("RGB", (1000, 1000), "white")
            
    bg_scaled_size = int(1000 * scale_ratio)
    resized_bg = bg_img.resize((bg_scaled_size, bg_scaled_size), Image.LANCZOS)

    final_canvas = Image.new("RGB", (image_size, image_size))
    
    for y in range(0, image_size, bg_scaled_size):
        for x in range(0, image_size, bg_scaled_size):
            final_canvas.paste(resized_bg, (x, y))

    strip_img = Image.new("RGB", (image_size, new_strip_height))
    current_x = 0
    
    for idx, img in enumerate(processed_imgs):
        new_w = int(img.width * scale_ratio)
        new_h = new_strip_height
        
        if idx == len(processed_imgs) - 1:
            new_w = image_size - current_x
        
        resized_char = img.resize((new_w, new_h), Image.LANCZOS)
        strip_img.paste(resized_char, (current_x, 0))
        current_x += new_w
    
    start_y = (image_size - new_strip_height) // 2
    final_canvas.paste(strip_img, (0, start_y))

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    final_canvas.save(output_path)

def main():
    
    if len(input_text) == 0:
        print("Input text is empty. No images will be generated.")
        return
    elif len(input_text) == 1:
        single(input_text)
    else:
        comb(input_text)

if __name__ == "__main__":
    main()