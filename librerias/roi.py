import cv2
import numpy as np

def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect

def calcular_dimensiones_roi(rect):
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    return maxWidth, maxHeight

def extraer_roi_plano(img_bgr, pts):
    """
    Recibe la imagen original y los 4 puntos.
    Devuelve la imagen recortada y aplanada, o None si los puntos son inválidos.
    """
    if len(pts) != 4:
        return None
        
    rect = order_points(pts)
    max_width, max_height = calcular_dimensiones_roi(rect)
    
    # Validación de ROI mínima
    if max_width < 10 or max_height < 10:
        return None

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    recorte_plano = cv2.warpPerspective(img_bgr, M, (max_width, max_height))
    
    return recorte_plano
