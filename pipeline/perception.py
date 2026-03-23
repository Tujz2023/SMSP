import cv2
import numpy as np
from PIL import Image, ImageFilter

def blur_and_histogram(image, blur_radius=17):
    image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    img_array = np.array(image)
    img_yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return Image.fromarray(img_output)
    
def filtered_image(image, blur_amount=61):
    image = np.array(image)[:, :, ::-1]
    blurred_image = cv2.GaussianBlur(image, (blur_amount, blur_amount), 0)
    blurred_image = cv2.blur(blurred_image, (20, 20))
    blurred_image = cv2.medianBlur(blurred_image, 5)
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-1,-2,-1],[-2,13,-2],[-1,-2,-1]])
    sharpened_iamge = cv2.filter2D(gray_image, -1, kernel)
    return Image.fromarray(sharpened_iamge)

def resize(image, target_content_size=(100, 100)):
    orig_w, orig_h = image.size
    a, b = target_content_size
    small_image = image.resize((a, b), Image.Resampling.LANCZOS)
    background = Image.new("RGB", (orig_w, orig_h), (255, 255, 255))
    offset_x = (orig_w - a) // 2
    offset_y = (orig_h - b) // 2
    background.paste(small_image, (offset_x, offset_y))
    return background

def fft_extract(image, radius_ratio=0.012, analysis='False'):
    img = np.array(image)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    radius = int(min(rows, cols) * radius_ratio)
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 1, -1)
    fshift_filtered = fshift * mask

    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    img_back_norm = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_back_enhanced = img_back_norm

    if analysis == 'False':
        return Image.fromarray(img_back_enhanced)
    return img, img_back_enhanced, fshift, fshift_filtered

def perception_module(image, percep_type, lambda_i=None, s_i=None):
    if percep_type == 'blur_and_histogram':
        return blur_and_histogram(image.convert("RGB"))
    elif percep_type == 'filtered_image':
        return filtered_image(image.convert("RGB"))
    else:
        fft_image = fft_extract(image.convert('L'), lambda_i)
        resized_image = resize(fft_image, (int(s_i), int(s_i)))
        return resized_image