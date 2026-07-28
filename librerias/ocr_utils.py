import cv2
import numpy as np
try:
    import easyocr
except ImportError:
    easyocr = None
try:
    import pytesseract
except ImportError:
    pytesseract = None
from utils.image_utils import asegurar_gris_uint8, resize_por_alto, unsharp_mask
from utils.segmentation import limpiar_strip_para_ocr
from utils.validation import limpiar_lectura_ocr, peso_fuente_ocr
from utils.config import ALLOWLIST_PLACA

def preparar_para_easyocr(img, alto_objetivo=128, fondo_blanco=True):
    gray = asegurar_gris_uint8(img)
    if gray is None or gray.size == 0:
        return None

    gray = resize_por_alto(gray, alto_objetivo)
    gray = unsharp_mask(gray, amount=0.8, sigma=1.0)

    # Si se pide fondo blanco y la imagen está invertida, corregimos polaridad.
    if fondo_blanco:
        # En placas/binarizados esperamos texto oscuro y fondo claro.
        if np.mean(gray) < 127:
            gray = cv2.bitwise_not(gray)

    gray = cv2.copyMakeBorder(gray, 20, 20, 30, 30, cv2.BORDER_CONSTANT, value=255)
    return gray

def normalizar_resultados_easyocr(resultados):
    normalizados = []
    for item in resultados:
        if isinstance(item, str):
            normalizados.append((None, item, 1.0))
        else:
            bbox = item[0]
            texto = item[1]
            conf = item[2] if len(item) > 2 else 1.0
            normalizados.append((bbox, texto, conf))
    return normalizados

def x_min_bbox(bbox):
    if bbox is None:
        return 0
    try:
        return min(p[0] for p in bbox)
    except Exception:
        return 0

def crear_item_lectura(texto, fuente, conf=1.0):
    return {
        "texto": limpiar_lectura_ocr(texto),
        "fuente": fuente,
        "conf": float(conf or 0.0),
        "peso": peso_fuente_ocr(fuente),
    }

def leer_easyocr_robusto(reader, img, nombre=""):
    lecturas = []
    if reader is None:
        return lecturas
    if img is None or img.size == 0:
        return lecturas

    # En la tira global conviene usar más altura porque allí ya está limpio el texto.
    alto = 170 if "strip" in str(nombre).lower() else 128
    img_ocr = preparar_para_easyocr(img, alto_objetivo=alto, fondo_blanco=True)
    if img_ocr is None or img_ocr.size == 0:
        return lecturas

    # Variantes internas: original normalizada, más contraste y umbralizada.
    variantes = [(nombre, img_ocr)]

    if "strip" in str(nombre).lower():
        try:
            # Variante limpia original (puede retener algo de borde)
            strip_clean = limpiar_strip_para_ocr(img, remover_bordes_laterales=False)
            if strip_clean is not None and strip_clean.size > 0:
                variantes.insert(0, (nombre + "_con_borde", strip_clean))
                
            # Variante sin bordes laterales (nuevo)
            strip_clean_no_bordes = limpiar_strip_para_ocr(img, remover_bordes_laterales=True)
            if strip_clean_no_bordes is not None and strip_clean_no_bordes.size > 0:
                variantes.insert(0, (nombre + "_sin_bordes_laterales", strip_clean_no_bordes))

            g = asegurar_gris_uint8(img_ocr)
            g_blur = cv2.GaussianBlur(g, (3, 3), 0)
            _, th = cv2.threshold(g_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Adelgazado ligero para evitar que 5/8/L/P se vuelvan manchas.
            kernel = np.ones((2, 2), np.uint8)
            th_thin = cv2.dilate(th, kernel, iterations=1)

            variantes.append((nombre + "_otsu", th))
            variantes.append((nombre + "_thin", th_thin))
            variantes.append((nombre + "_big", resize_por_alto(g, 220)))
        except Exception:
            pass

    for nombre_var, img_var in variantes:
        try:
            resultados = reader.readtext(
                img_var,
                detail=1,
                paragraph=False,
                allowlist=ALLOWLIST_PLACA,
                decoder="beamsearch",
                mag_ratio=1.8,
                text_threshold=0.20,
                low_text=0.20,
                link_threshold=0.20,
                contrast_ths=0.05,
                adjust_contrast=0.70,
                width_ths=1.20,
                add_margin=0.08,
            )
        except TypeError:
            resultados = reader.readtext(
                img_var,
                detail=1,
                paragraph=False,
                allowlist=ALLOWLIST_PLACA,
            )
        except Exception as e:
            print(f"[WARN OCR {nombre_var}] EasyOCR falló: {e}")
            continue

        normalizados = normalizar_resultados_easyocr(resultados)

        # 1) Lecturas individuales.
        for bbox, texto, conf in normalizados:
            texto_limpio = limpiar_lectura_ocr(texto)
            if len(texto_limpio) >= 2:
                lecturas.append(crear_item_lectura(texto_limpio, nombre_var, conf))
                print(f"[OCR {nombre_var}] {texto_limpio} | conf={conf:.2f}")

        # 2) Lectura unida de izquierda a derecha.
        if len(normalizados) >= 2:
            ordenados = sorted(normalizados, key=lambda x: x_min_bbox(x[0]))
            unido = "".join(limpiar_lectura_ocr(x[1]) for x in ordenados)
            unido = limpiar_lectura_ocr(unido)
            if len(unido) >= 5:
                conf_prom = sum(float(x[2] or 0.0) for x in ordenados) / max(1, len(ordenados))
                lecturas.append(crear_item_lectura(unido, nombre_var + "_unido", conf_prom))
                print(f"[OCR {nombre_var} UNIDO] {unido}")

    return lecturas

def leer_tesseract_robusto(img, nombre="tesseract"):
    lecturas = []
    if pytesseract is None or img is None or getattr(img, "size", 0) == 0:
        return lecturas

    img_ocr = preparar_para_easyocr(img, alto_objetivo=170 if "strip" in str(nombre).lower() else 140, fondo_blanco=True)
    if img_ocr is None or img_ocr.size == 0:
        return lecturas

    variantes = [(nombre, img_ocr)]
    try:
        g = asegurar_gris_uint8(img_ocr)
        _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variantes.append((nombre + "_otsu", th))
    except Exception:
        pass

    config = (
        "--oem 3 --psm 7 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    for fuente, im in variantes:
        try:
            data = pytesseract.image_to_data(
                im,
                config=config,
                output_type=pytesseract.Output.DICT,
            )

            textos = []
            confs = []
            for txt, conf in zip(data.get("text", []), data.get("conf", [])):
                limpio = limpiar_lectura_ocr(txt)
                try:
                    c = float(conf)
                except Exception:
                    c = -1.0
                if limpio:
                    textos.append(limpio)
                    if c >= 0:
                        confs.append(c / 100.0)

            unido = limpiar_lectura_ocr("".join(textos))
            if len(unido) >= 2:
                conf_prom = float(np.mean(confs)) if confs else 0.50
                lecturas.append(crear_item_lectura(unido, fuente, conf_prom))
                print(f"[OCR {fuente}] {unido} | conf={conf_prom:.2f}")

        except Exception as e:
            print(f"[WARN OCR {fuente}] Tesseract falló: {e}")

    return lecturas

def agregar_variante(lista, nombre, img):
    if img is not None and getattr(img, "size", 0) > 0:
        lista.append((nombre, img))

