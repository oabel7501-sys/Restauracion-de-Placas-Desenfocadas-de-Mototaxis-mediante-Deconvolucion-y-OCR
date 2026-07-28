import cv2
import numpy as np
from utils.config import (
    PSF_TIPO, MOTION_BLUR_LENGTH, MOTION_BLUR_ANGLE, DEFOCUS_RADIUS,
    WIENER_K, EPSILON_FFT, POST_BILATERAL, POST_CLAHE, POST_SHARPEN,
    SAFE_RESTORE_SELECTION
)
from utils.image_utils import asegurar_gris_uint8, unsharp_mask

def crear_psf_motion(size=17, angle=0.0):
    size = int(max(3, size))
    if size % 2 == 0:
        size += 1

    psf = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    cv2.line(psf, (0, center), (size - 1, center), 1.0, 1)

    if abs(float(angle)) > 1e-6:
        M = cv2.getRotationMatrix2D((center, center), float(angle), 1.0)
        psf = cv2.warpAffine(psf, M, (size, size), flags=cv2.INTER_CUBIC)

    psf_sum = float(np.sum(psf))
    if psf_sum > 0:
        psf /= psf_sum
    else:
        psf[center, center] = 1.0
    return psf

def crear_psf_defocus(radius=5):
    radius = int(max(1, radius))
    size = radius * 2 + 1
    psf = np.zeros((size, size), dtype=np.float32)
    cv2.circle(psf, (radius, radius), radius, 1.0, -1)
    psf_sum = float(np.sum(psf))
    if psf_sum > 0:
        psf /= psf_sum
    else:
        psf[radius, radius] = 1.0
    return psf

def pad_psf_to_image(psf, shape):
    h, w = shape[:2]
    padded = np.zeros((h, w), dtype=np.float32)
    kh, kw = psf.shape[:2]
    padded[:kh, :kw] = psf
    padded = np.fft.ifftshift(padded)
    return padded

def wiener_deconvolution_manual(gray, psf, k=0.006, eps=1e-8, padding=True):
    gray = asegurar_gris_uint8(gray)
    if gray is None or gray.size == 0:
        return gray

    img = gray.astype(np.float32) / 255.0

    if padding:
        ph = max(8, min(40, img.shape[0] // 6))
        pw = max(8, min(40, img.shape[1] // 6))
        img_pad = cv2.copyMakeBorder(img, ph, ph, pw, pw, cv2.BORDER_REFLECT_101)
    else:
        ph = pw = 0
        img_pad = img

    psf_pad = pad_psf_to_image(psf, img_pad.shape)
    G = np.fft.fft2(img_pad)
    H = np.fft.fft2(psf_pad)
    H_conj = np.conj(H)

    F_hat = (H_conj / (np.abs(H) ** 2 + float(k) + float(eps))) * G
    restored = np.real(np.fft.ifft2(F_hat))

    if padding:
        restored = restored[ph:ph + img.shape[0], pw:pw + img.shape[1]]

    restored = np.clip(restored, 0.0, 1.0)
    return (restored * 255.0).astype(np.uint8)

def aplicar_postprocesamiento_visual(gray):
    out = asegurar_gris_uint8(gray)
    if out is None:
        return None

    if POST_BILATERAL:
        out = cv2.bilateralFilter(out, 7, 45, 45)

    if POST_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        out = clahe.apply(out)

    if POST_SHARPEN:
        out = unsharp_mask(out, amount=0.85, sigma=1.0)

    return out

def varianza_laplaciano(gray):
    g = asegurar_gris_uint8(gray)
    if g is None or g.size == 0:
        return 0.0
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

def restaurar_imagen_borrosa(
    img_bgr,
    psf_tipo=PSF_TIPO,
    motion_length=MOTION_BLUR_LENGTH,
    motion_angle=MOTION_BLUR_ANGLE,
    defocus_radius=DEFOCUS_RADIUS,
    wiener_k=WIENER_K,
):
    gray = asegurar_gris_uint8(img_bgr)
    if gray is None or gray.size == 0:
        return gray, gray, gray, gray

    clahe_vis = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    sharp_clahe = unsharp_mask(clahe_vis, amount=0.65, sigma=1.0)

    if str(psf_tipo).lower() == "defocus":
        psf = crear_psf_defocus(defocus_radius)
    else:
        psf = crear_psf_motion(motion_length, motion_angle)

    deconv = wiener_deconvolution_manual(gray, psf, k=wiener_k, eps=EPSILON_FFT, padding=True)
    deconv_post = aplicar_postprocesamiento_visual(deconv)
    sharp_post = unsharp_mask(deconv_post, amount=0.75, sigma=1.0)

    if SAFE_RESTORE_SELECTION:
        # Importado aquí localmente si es necesario, pero como hemos sacado la función:
        from utils.segmentation import calidad_imagen_para_ocr
        candidatos = [
            ("gray", gray),
            ("clahe", clahe_vis),
            ("clahe_sharp", sharp_clahe),
            ("wiener", deconv),
            ("wiener_post", deconv_post),
            ("wiener_post_sharp", sharp_post),
        ]
        scores = [(nombre, calidad_imagen_para_ocr(im)) for nombre, im in candidatos]
        nombre_best, _ = max(scores, key=lambda x: x[1])
        img_restaurada = dict(candidatos)[nombre_best]
        print("[DEBUG RESTAURACION] Calidad candidatos:", [(n, round(v, 1)) for n, v in scores], "=>", nombre_best)
    else:
        img_restaurada = deconv_post

    img_sharp = unsharp_mask(img_restaurada, amount=0.80, sigma=1.0)
    return img_restaurada, clahe_vis, gray, img_sharp
