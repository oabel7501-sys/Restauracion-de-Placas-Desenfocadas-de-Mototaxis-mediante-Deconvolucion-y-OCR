import cv2
import numpy as np

def asegurar_gris_uint8(img):
    if img is None:
        return None

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray

def unsharp_mask(gray, amount=1.2, sigma=1.0):
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    sharp = cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def normalizar_uint8(img):
    if img is None:
        return None
    arr = img.astype(np.float32)
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - mn) / (mx - mn)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)

def resize_por_alto(img, alto_objetivo=120):
    if img is None or img.size == 0:
        return img

    h, w = img.shape[:2]
    if h <= 0:
        return img

    escala = alto_objetivo / float(h)
    nuevo_w = max(1, int(w * escala))
    return cv2.resize(img, (nuevo_w, alto_objetivo), interpolation=cv2.INTER_CUBIC)


def aplicar_gamma(gray, gamma=0.70):
    g = asegurar_gris_uint8(gray)
    if g is None or g.size == 0:
        return g
    gamma = max(0.20, float(gamma))
    table = ((np.arange(256) / 255.0) ** gamma * 255.0).astype(np.uint8)
    return cv2.LUT(g, table)

def mejorar_baja_luz(gray):
    g = asegurar_gris_uint8(gray)
    if g is None or g.size == 0:
        return g
    g = aplicar_gamma(g, gamma=0.62)
    clahe = cv2.createCLAHE(clipLimit=3.2, tileGridSize=(8, 8))
    g = clahe.apply(g)
    g = unsharp_mask(g, amount=0.55, sigma=1.0)
    return g

def clahe_fuerte(gray):
    g = asegurar_gris_uint8(gray)
    if g is None or g.size == 0:
        return g
    clahe = cv2.createCLAHE(clipLimit=3.8, tileGridSize=(8, 8))
    return clahe.apply(g)