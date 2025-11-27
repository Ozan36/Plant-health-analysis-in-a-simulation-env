import cv2
import numpy as np


def apply_band_filter(img, band, verbose=False):
 
    # Hyperspectral band filtresi uygular
  
    b = int(max(0, min(223, band)))
    I = img.astype(np.float32) / 255.0

    if b < 30:
        f = b / 30.0
        I[:,:,0] = np.clip(I[:,:,0] * (1.8 - 0.8*f), 0, 1)
        I[:,:,1] = np.clip(I[:,:,1] * (0.3 + 0.5*f), 0, 1)
        I[:,:,2] = np.clip(I[:,:,2] * (0.2 + 0.3*f), 0, 1)
    elif b < 60:
        f = (b - 30) / 30.0
        I[:,:,0] = np.clip(I[:,:,0] * (1.0 + 1.0*f), 0, 1)
        I[:,:,1] = np.clip(I[:,:,1] * (0.8 + 0.4*f), 0, 1)
        I[:,:,2] = np.clip(I[:,:,2] * (0.5 + 0.3*f), 0, 1)
    elif b < 90:
        f = (b - 60) / 30.0
        I[:,:,0] = np.clip(I[:,:,0] * (2.0 - 0.5*f), 0, 1)
        I[:,:,1] = np.clip(I[:,:,1] * (1.2 + 0.6*f), 0, 1)
        I[:,:,2] = np.clip(I[:,:,2] * (0.8 + 0.4*f), 0, 1)
    elif b < 120:
        f = (b - 90) / 30.0
        I[:,:,0] = np.clip(I[:,:,0] * (1.5 - 0.8*f), 0, 1)
        I[:,:,1] = np.clip(I[:,:,1] * (1.8 - 0.3*f), 0, 1)
        I[:,:,2] = np.clip(I[:,:,2] * (1.2 - 0.4*f), 0, 1)
    elif b < 150:
        f = (b - 120) / 30.0
        I[:,:,0] = np.clip(I[:,:,0] * (0.7 - 0.4*f), 0, 1)
        I[:,:,1] = np.clip(I[:,:,1] * (1.5 + 0.3*f), 0, 1)
        I[:,:,2] = np.clip(I[:,:,2] * (1.5 + 0.5*f), 0, 1)
    elif b < 180:
        f = (b - 150) / 30.0
        I[:,:,0] = np.clip(I[:,:,0] * (0.3 - 0.2*f), 0, 1)
        I[:,:,1] = np.clip(I[:,:,1] * (1.8 - 0.8*f), 0, 1)
        I[:,:,2] = np.clip(I[:,:,2] * (2.0 - 0.3*f), 0, 1)
    else:
        f = (b - 180) / 43.0
        I[:,:,2] = np.clip(I[:,:,2] * (0.5 + 2.0*f), 0, 1)
        I[:,:,1] = np.clip(I[:,:,1] * (1.0 + 1.0*f), 0, 1)
        I[:,:,0] = np.clip(I[:,:,0] * (0.1 + 0.5*f), 0, 1)
    
    out = (I * 255).astype(np.uint8)
    
    # Kontrast iyileştirme
    try:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, 1.2)
        v = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(v)
        hsv = cv2.merge([h, np.clip(s, 0, 255), v])
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    except Exception as e:
        if verbose:
            print(f'Kontrast iyileştirme hatası: {e}')
    
    return out


def create_simple_green_mask(img):
    """
    HSV renk aralığına göre yeşil maske oluşturur
    
    Args:
        img: BGR görüntü
    
    Returns:
        Binary maske
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (40, 60, 60), (85, 255, 255))
    
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
    
    return mask


def clean_mask(mask, min_area=500):
    """
    Maskeyi temizler ve küçük alanları kaldırır
    
    Args:
        mask: Binary maske
        min_area: Minimum alan boyutu
    
    Returns:
        Temizlenmiş maske
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)
    
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(clean, [cnt], -1, 255, -1)
    
    return clean


def apply_mask_to_image(img, mask):
 
    white = np.ones_like(img) * 255
    masked = cv2.bitwise_and(img, img, mask=mask)
    masked = cv2.add(masked, cv2.bitwise_and(white, white, mask=cv2.bitwise_not(mask)))
    return masked

 # Hyperspectral band aralıkları
def get_band_type(band):
    """Band tipini döndürür"""
    if band < 30:
        return "UV/Morötesi"
    elif band < 60:
        return "Mavi"
    elif band < 90:
        return "Cyan"
    elif band < 120:
        return "Yeşil"
    elif band < 150:
        return "Sarı/Turuncu"
    elif band < 180:
        return "Kırmızı"
    else:
        return "NIR/Kızılötesi"


