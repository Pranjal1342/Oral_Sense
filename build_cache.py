import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image, ImageOps, ExifTags
import tqdm
import hashlib

DATA_DIR = 'data/wli'
CACHE_ROI_DIR = 'data_cached_roi'
CACHE_CENTER_DIR = 'data_cached_center'
SIZES = [256, 128, 96]
WORKING_SIZE = 256

for d in [CACHE_ROI_DIR, CACHE_CENTER_DIR]:
    for size in SIZES:
        for split in ['cancer', 'non_cancer']:
            os.makedirs(os.path.join(d, f'{size}x{size}', split), exist_ok=True)

gamma = 1.4
lut = ((np.arange(256) / 255.0) ** (1.0 / gamma) * 255).astype(np.uint8)
EPS = 1e-6
qhash_cache = {}

def get_exif_model_hash(img):
    cam = 'UNKNOWN'
    try:
        exif = img.getexif()
        if exif:
            for k, v in exif.items():
                if ExifTags.TAGS.get(k) == 'Model':
                    cam = str(v).strip()
    except: pass
    p = 'NONE'
    try:
        q = img.quantization
        if q:
            h_str = str(q.values())
            if h_str not in qhash_cache:
                qhash_cache[h_str] = hashlib.md5(h_str.encode('UTF-8')).hexdigest()[:8]
            p = qhash_cache[h_str]
    except: pass
    return cam, p

def to_pseudo_afi(bgr_img, lut, clahe_clip=2.0, shadow_thresh=25, specular_thresh=240, neutral_value=128, denoise_ksize=3):
    if bgr_img is None or bgr_img.ndim != 3 or bgr_img.shape[2] != 3 or bgr_img.size == 0:
        return None, None
    bgr = bgr_img.astype(np.float32)
    b, g, r = cv2.split(bgr)
    max_ch = np.maximum(np.maximum(b, g), r)
    min_ch = np.minimum(np.minimum(b, g), r)
    unreliable_mask = (max_ch < shadow_thresh) | (min_ch > specular_thresh)
    mask_pct = 100.0 * float(unreliable_mask.mean())
    ratio = g / (r + EPS)
    ratio = np.clip(ratio, 0, 3.0)
    ratio_norm = (ratio / 3.0 * 255.0).astype(np.uint8)
    if denoise_ksize and denoise_ksize > 1:
        ratio_norm = cv2.medianBlur(ratio_norm, denoise_ksize)
    ratio_gamma = cv2.LUT(ratio_norm, lut)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    result = clahe.apply(ratio_gamma)
    result[unreliable_mask] = neutral_value
    return result, mask_pct

def detect_roi_bbox_saturation(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    s = hsv[:,:,1]
    _, thresh = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    img_h, img_w = img_rgb.shape[:2]
    k_size = max(3, int(0.02 * min(img_h, img_w)))
    if k_size % 2 == 0: k_size += 1
    
    kernel = np.ones((k_size, k_size), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, True, 0, 'no_contours'
        
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    pad_w = int(w * 0.08)
    pad_h = int(h * 0.08)
    x = max(0, x - pad_w)
    y = max(0, y - pad_h)
    w = min(img_w - x, w + 2*pad_w)
    h = min(img_h - y, h + 2*pad_h)
    
    area_frac = (w * h) / (img_w * img_h)
    
    reason = ''
    if area_frac < 0.10: reason = 'too_small'
    elif area_frac > 0.95: reason = 'too_large'
    low_conf = reason != ''
    
    return (x, y, w, h), low_conf, area_frac, reason

def make_square_box(x, y, w, h, img_w, img_h):
    center_x, center_y = x + w // 2, y + h // 2
    side = max(w, h)
    if side > img_w or side > img_h:
        side = min(img_w, img_h)
    x0 = max(0, center_x - side // 2)
    y0 = max(0, center_y - side // 2)
    if x0 + side > img_w: x0 = img_w - side
    if y0 + side > img_h: y0 = img_h - side
    return x0, y0, side

def extract_features(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    means = np.mean(img_np, axis=(0,1))
    stds = np.std(img_np, axis=(0,1))
    return lap_var, means, stds

def process_class(label_str, label_int):
    wli_in = os.path.join(DATA_DIR, label_str)
    files = [f for f in os.listdir(wli_in) if not f.lower().startswith('cancer_aug_')]
    
    manifest = []
    
    for f in tqdm.tqdm(files, desc=f'Processing {label_str}'):
        w_path = os.path.join(wli_in, f)
        base, _ = os.path.splitext(f)
            
        try:
            with Image.open(w_path) as w_img:
                w_img = ImageOps.exif_transpose(w_img)
                w, h = w_img.size
                cam, q = get_exif_model_hash(w_img)
                cluster_id = f"{cam}_{round(w/100)*100}x{round(h/100)*100}_{q}"
                
                w_np = np.array(w_img.convert('RGB'))
                img_h, img_w = w_np.shape[:2]
                
                # ROI Crop
                bbox, low_conf, area_frac, fallback_reason = detect_roi_bbox_saturation(w_np)
                if bbox is None or low_conf:
                    side_roi = min(img_w, img_h)
                    x0_roi, y0_roi = (img_w - side_roi) // 2, (img_h - side_roi) // 2
                else:
                    x0_roi, y0_roi, side_roi = make_square_box(*bbox, img_w, img_h)
                
                # Center Crop (uniform)
                side_center = min(img_w, img_h)
                x0_center, y0_center = (img_w - side_center) // 2, (img_h - side_center) // 2
                
                # Process ROI
                w_crop_roi = w_np[y0_roi:y0_roi+side_roi, x0_roi:x0_roi+side_roi]
                w_crop_256_roi = cv2.resize(w_crop_roi, (256, 256), interpolation=cv2.INTER_AREA)
                w_crop_256_roi_bgr = cv2.cvtColor(w_crop_256_roi, cv2.COLOR_RGB2BGR)
                a_crop_256_roi, mask_pct_roi = to_pseudo_afi(w_crop_256_roi_bgr, lut)
                
                # Process Center
                w_crop_center = w_np[y0_center:y0_center+side_center, x0_center:x0_center+side_center]
                w_crop_256_center = cv2.resize(w_crop_center, (256, 256), interpolation=cv2.INTER_AREA)
                w_crop_256_center_bgr = cv2.cvtColor(w_crop_256_center, cv2.COLOR_RGB2BGR)
                a_crop_256_center, _ = to_pseudo_afi(w_crop_256_center_bgr, lut)
                
                lap_var, means, stds = extract_features(w_crop_256_roi)
                
                for size in SIZES:
                    # ROI save
                    w_res_r = cv2.resize(w_crop_256_roi, (size, size), interpolation=cv2.INTER_AREA)
                    a_res_r = cv2.resize(a_crop_256_roi, (size, size), interpolation=cv2.INTER_AREA)
                    np.savez_compressed(os.path.join(CACHE_ROI_DIR, f'{size}x{size}', label_str, f'{base}.npz'), wli=w_res_r, afi=a_res_r)
                    
                    # Center save
                    w_res_c = cv2.resize(w_crop_256_center, (size, size), interpolation=cv2.INTER_AREA)
                    a_res_c = cv2.resize(a_crop_256_center, (size, size), interpolation=cv2.INTER_AREA)
                    np.savez_compressed(os.path.join(CACHE_CENTER_DIR, f'{size}x{size}', label_str, f'{base}.npz'), wli=w_res_c, afi=a_res_c)
                
                manifest.append({
                    'file': f,
                    'class': label_int,
                    'cluster_id': cluster_id,
                    'phash_group': 'TODO',
                    'roi_area_frac': area_frac,
                    'fallback': low_conf,
                    'fallback_reason': fallback_reason,
                    'crop_side': side_roi,
                    'mask_pct': mask_pct_roi,
                    'laplacian_var': lap_var,
                    'mean_r': means[0], 'mean_g': means[1], 'mean_b': means[2],
                    'std_r': stds[0], 'std_g': stds[1], 'std_b': stds[2]
                })
        except Exception as e:
            pass
            
    return manifest

if __name__ == '__main__':
    m_c = process_class('cancer', 1)
    m_h = process_class('non_cancer', 0)
    df = pd.DataFrame(m_c + m_h)
    df.to_csv('cache_manifest.csv', index=False)
