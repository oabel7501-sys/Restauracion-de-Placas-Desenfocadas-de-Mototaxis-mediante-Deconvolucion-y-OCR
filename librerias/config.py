# utils/config.py

# ===================================================================
# CONFIGURACIÓN GENERAL
# ===================================================================
ALLOWLIST_PLACA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
DEBUG_GUARDAR_IMAGENES = True

# Parámetros base del modelo de restauración.
# Pueden ajustarse en pruebas reales según el desenfoque observado.
PSF_TIPO = "motion"          # "motion" o "defocus"
MOTION_BLUR_LENGTH = 17      # longitud del desenfoque de movimiento, en píxeles
MOTION_BLUR_ANGLE = 0.0      # ángulo del desenfoque en grados
DEFOCUS_RADIUS = 5           # radio del disco para desenfoque óptico
WIENER_K = 0.006             # relación ruido/señal. Menor = más agresivo.
EPSILON_FFT = 1e-8

# Postprocesamiento visual posterior a la deconvolución.
POST_BILATERAL = True
POST_CLAHE = True
POST_SHARPEN = True
SAFE_RESTORE_SELECTION = True  # evita que Wiener agresivo gane si genera artefactos

# Parámetros para detección automática de placas celestes (Motos)
HSV_LOWER_CELESTE = [80, 30, 80]
HSV_UPPER_CELESTE = [105, 255, 255]
AUTO_CROP_MARGIN = 16
MIN_PLATE_AREA_RATIO = 0.0003
MAX_PLATE_AREA_RATIO = 0.05
PLATE_ASPECT_MIN = 1.4
PLATE_ASPECT_MAX = 4.5
MIN_TEXT_COMPONENTS_AUTO = 3
MIN_TEXT_COMPONENTS_DUDOSO = 2
MIN_DARK_RATIO_AUTO = 0.015
MAX_DARK_RATIO_AUTO = 0.45
