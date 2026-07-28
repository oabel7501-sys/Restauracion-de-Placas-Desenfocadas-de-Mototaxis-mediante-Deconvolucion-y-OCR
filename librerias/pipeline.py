from utils.ocr_utils import crear_item_lectura
from utils.validation import limpiar_lectura_ocr, normalizar_lectura_item
import cv2
import numpy as np
from utils.restoration import restaurar_imagen_borrosa
from utils.segmentation import segmentar_y_leer, recortar_zona_caracteres
from utils.validation import validar_pool_lecturas
from utils.ocr_utils import leer_easyocr_robusto, leer_tesseract_robusto, agregar_variante
from utils.debug_tools import guardar_debug_pipeline
from utils.image_utils import mejorar_baja_luz, clahe_fuerte, unsharp_mask
from utils.config import DEBUG_GUARDAR_IMAGENES

def procesar_fragmento(img, reader, is_celeste=False):
    if img is None or img.size == 0:
        return {"valido": False, "tipo": "Imagen vacía", "placa": ""}

    # Si la ROI es extremadamente pequeña, se amplía moderadamente para que OCR tenga más píxeles.
    # La deconvolución sigue trabajando solo sobre la placa, no sobre toda la escena.
    h0, w0 = img.shape[:2]
    if h0 < 80:
        escala = 80.0 / max(1.0, float(h0))
        img = cv2.resize(img, (max(1, int(w0 * escala)), 80), interpolation=cv2.INTER_CUBIC)

    img_restaurada, img_clahe, img_gray, img_sharp = restaurar_imagen_borrosa(img)

    img_bin, img_contornos, caracteres, img_strip, bins = segmentar_y_leer(img_restaurada, img_clahe, reader, img_bgr=img, is_celeste=is_celeste)

    h_img = img_restaurada.shape[0]
    zona_gray = recortar_zona_caracteres(img_gray)
    zona_clahe = recortar_zona_caracteres(img_clahe)
    zona_rest = recortar_zona_caracteres(img_restaurada)
    zona_sharp = recortar_zona_caracteres(img_sharp)
    zona_lowlight = mejorar_baja_luz(zona_gray)
    zona_clahe_fuerte = clahe_fuerte(zona_gray)
    zona_sharp_fuerte = unsharp_mask(zona_clahe_fuerte, amount=0.95, sigma=1.0) if zona_clahe_fuerte is not None else None
    # bins ya corresponden a la zona de caracteres; no recortar otra vez.
    zona_otsu = bins["otsu"]
    zona_adapt = bins["adapt"]
    zona_otsu_inv = bins["otsu_inv"]
    zona_adapt_inv = bins["adapt_inv"]
    zona_base_segmentacion = bins.get("zona_base", zona_rest)

    variantes_ocr = []
    agregar_variante(variantes_ocr, "gray_completa", img_gray)
    agregar_variante(variantes_ocr, "clahe_completa", img_clahe)
    agregar_variante(variantes_ocr, "restaurada_completa", img_restaurada)
    agregar_variante(variantes_ocr, "sharp_completa", img_sharp)

    agregar_variante(variantes_ocr, "zona_gray", zona_gray)
    agregar_variante(variantes_ocr, "zona_clahe", zona_clahe)
    agregar_variante(variantes_ocr, "zona_restaurada", zona_rest)
    agregar_variante(variantes_ocr, "zona_sharp", zona_sharp)
    agregar_variante(variantes_ocr, "zona_lowlight", zona_lowlight)
    agregar_variante(variantes_ocr, "zona_clahe_fuerte", zona_clahe_fuerte)
    agregar_variante(variantes_ocr, "zona_sharp_fuerte", zona_sharp_fuerte)
    agregar_variante(variantes_ocr, "zona_base_segmentacion", zona_base_segmentacion)
    agregar_variante(variantes_ocr, "zona_otsu", zona_otsu)
    agregar_variante(variantes_ocr, "zona_adapt", zona_adapt)
    agregar_variante(variantes_ocr, "zona_otsu_inv", cv2.bitwise_not(zona_otsu_inv))
    agregar_variante(variantes_ocr, "zona_adapt_inv", cv2.bitwise_not(zona_adapt_inv))
    agregar_variante(variantes_ocr, "strip_caracteres", img_strip)
    
    if "letras_negras" in bins:
        agregar_variante(variantes_ocr, "zona_letras_negras", bins["letras_negras"])
    if "letras_negras_inv" in bins:
        agregar_variante(variantes_ocr, "zona_letras_negras_inv", cv2.bitwise_not(bins["letras_negras_inv"]))
    if "strip_letras_negras" in bins:
        agregar_variante(variantes_ocr, "strip_letras_negras", bins["strip_letras_negras"])
    if "strip_letras_negras_suave" in bins:
        agregar_variante(variantes_ocr, "strip_letras_negras_suave", bins["strip_letras_negras_suave"])
    if "zona_gray_celeste" in bins:
        agregar_variante(variantes_ocr, "zona_gray_celeste", bins["zona_gray_celeste"])
    if "zona_clahe_celeste" in bins:
        agregar_variante(variantes_ocr, "zona_clahe_celeste", bins["zona_clahe_celeste"])
    if "canal_v_celeste" in bins:
        agregar_variante(variantes_ocr, "canal_v_celeste", bins["canal_v_celeste"])

    if DEBUG_GUARDAR_IMAGENES:
        try:
            cv2.imwrite("debug_roi_gray.jpg", img_gray)
            cv2.imwrite("debug_zona_caracteres.jpg", zona_sharp)
            cv2.imwrite("debug_bin_otsu_panel.jpg", img_bin)
            if img_strip is not None:
                cv2.imwrite("debug_strip_caracteres.jpg", img_strip)
        except Exception as e:
            print(f"[WARN] No se pudieron guardar imágenes debug: {e}")

    textos_crudos = []
    for nombre, imagen_ocr in variantes_ocr:
        # OCR clásico principal/compatible si Tesseract está instalado.
        textos_crudos.extend(leer_tesseract_robusto(imagen_ocr, "tess_" + nombre))
        # EasyOCR queda como motor robusto adicional/comparativo.
        textos_crudos.extend(leer_easyocr_robusto(reader, imagen_ocr, nombre))

    # IMPORTANTE:
    # textos_crudos contiene dicts con texto/fuente/confianza. No hay que convertirlos
    # con str(dict), porque eso mete palabras como TEXTO/FUENTE/CONF en la validación.
    lecturas_para_validar = list(textos_crudos)

    textos_unicos_debug = []
    textos_unicos_para_combinar = []
    for item in textos_crudos:
        texto, fuente, conf, peso = normalizar_lectura_item(item)
        if texto and texto not in textos_unicos_debug:
            textos_unicos_debug.append(f"{texto} [{fuente}]")
            textos_unicos_para_combinar.append(texto)

    # No unir todo como verdad principal. Solo una lectura secundaria de baja prioridad.
    if textos_unicos_para_combinar:
        combinado_corto = "".join(textos_unicos_para_combinar[:3])
        combinado_corto = limpiar_lectura_ocr(combinado_corto)
        if len(combinado_corto) >= 6:
            lecturas_para_validar.append(crear_item_lectura(combinado_corto, "combinado_baja_prioridad", 0.30))
            textos_unicos_debug.append(f"{combinado_corto} [combinado_baja_prioridad]")

    print("\n[DEBUG OCR] Lecturas crudas normalizadas:")
    print(textos_unicos_debug)

    if is_celeste and DEBUG_GUARDAR_IMAGENES:
        try:
            with open("results/debug/ocr_variantes_celeste.txt", "w", encoding="utf-8") as f:
                f.write("Variantes OCR para modo celeste:\n\n")
                for td in textos_unicos_debug:
                    f.write(f"{td}\n")
        except Exception as e:
            print(f"[WARN] No se pudo guardar ocr_variantes_celeste.txt: {e}")

    valido, placa, tipo, score, candidatos = validar_pool_lecturas(lecturas_para_validar, is_celeste=is_celeste)

    print("\n[DEBUG OCR] --- RESULTADO FINAL ---")
    if valido:
        print(f"GANADOR : {placa}")
        print(f"FORMATO : {tipo}")
        print(f"PUNTAJE : {score:.1f}")
    else:
        print("GANADOR : Ninguno")
        print(f"MOTIVO  : {tipo}")
        print(f"MEJOR APROX.: {placa}")
    print("-----------------------------------\n")

    return {
        "roi": img,
        "preprocesada": img_clahe,
        "binarizada": img_bin,
        "segmentacion": img_contornos,
        "caracteres": caracteres,
        "placa": placa,
        "valido": valido,
        "tipo": tipo,
        "score": score,
    }

