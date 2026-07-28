# utils/image_io.py
import cv2
import numpy as np

try:
    import rawpy
except ImportError:
    rawpy = None


def cargar_imagen(ruta_imagen):
    """
    Carga una imagen soportando varios formatos, incluyendo RAW (ARW) de forma opcional.
    """
    if str(ruta_imagen).lower().endswith(".arw"):
        if rawpy is None:
            raise RuntimeError("rawpy no está instalado. Instala con: pip install rawpy")
        with rawpy.imread(ruta_imagen) as raw:
            img = raw.postprocess(use_camera_wb=True)
            # Convertir de RGB a BGR para OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
    else:
        # Usamos imdecode para manejar rutas con caracteres especiales (ñ, tildes) en Windows
        with open(ruta_imagen, "rb") as f:
            chunk = f.read()
        chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
        img = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"No se pudo cargar la imagen: {ruta_imagen}")
        return img
