import cv2
import numpy as np
import os
from utils.image_utils import asegurar_gris_uint8
from utils.config import DEBUG_GUARDAR_IMAGENES

def varianza_laplaciano(gray):
    g = asegurar_gris_uint8(gray)
    if g is None or g.size == 0:
        return 0.0
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

def evaluar_binarizacion_inv(binary_inv):
    if binary_inv is None or binary_inv.size == 0:
        return -9999.0

    b = (binary_inv > 0).astype(np.uint8)
    h, w = b.shape[:2]
    if h < 10 or w < 20:
        return -9999.0

    fg_ratio = float(np.mean(b))
    score = 0.0

    # Una placa binarizada correctamente suele tener tinta moderada, no media imagen blanca.
    if 0.035 <= fg_ratio <= 0.34:
        score += 70.0 - abs(fg_ratio - 0.16) * 180.0
    else:
        score -= 120.0 + abs(fg_ratio - 0.16) * 120.0

    # Penalizar rayas que cruzan casi toda la banda.
    row_density = np.mean(b, axis=1)
    col_density = np.mean(b, axis=0)
    heavy_rows = float(np.mean(row_density > 0.62))
    heavy_cols = float(np.mean(col_density > 0.62))
    score -= (heavy_rows + heavy_cols) * 180.0

    # Componentes con tamaño compatible con caracteres.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    valid_cc = 0
    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        if area < h * w * 0.001:
            continue
        if bh < h * 0.18 or bh > h * 0.95:
            continue
        ratio = bw / float(bh) if bh else 99
        if 0.08 <= ratio <= 1.50:
            valid_cc += 1

    # Placas suelen tener alrededor de 6 caracteres; aceptamos rango amplio.
    if 3 <= valid_cc <= 10:
        score += 45.0
    else:
        score -= abs(valid_cc - 6) * 8.0

    return score

def binarizacion_robusta(gray):
    g = asegurar_gris_uint8(gray)
    if g is None or g.size == 0:
        return g, {}, {"nombre": "vacia", "score": -9999.0}

    # Ligero suavizado para estabilizar umbral, sin destruir bordes.
    suave = cv2.GaussianBlur(g, (3, 3), 0)
    variantes = {}

    _, otsu_inv = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, otsu = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adapt_inv = cv2.adaptiveThreshold(
        suave, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 9
    )
    adapt = cv2.bitwise_not(adapt_inv)

    variantes["otsu_inv"] = otsu_inv
    variantes["otsu"] = otsu
    variantes["adapt_inv"] = adapt_inv
    variantes["adapt"] = adapt

    # Normalizamos todas a texto blanco sobre fondo negro para evaluar y extraer strip.
    candidatos_inv = {
        "otsu_inv": otsu_inv,
        "otsu_to_inv": cv2.bitwise_not(otsu),
        "adapt_inv": adapt_inv,
        "adapt_to_inv": cv2.bitwise_not(adapt),
    }

    kernel = np.ones((2, 2), np.uint8)
    mejor_nombre = None
    mejor_img = None
    mejor_score = -99999.0

    for nombre, b in candidatos_inv.items():
        limpio = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel, iterations=1)
        limpio = limpiar_bordes_componentes_binary_inv(limpio)
        score = evaluar_binarizacion_inv(limpio)
        if score > mejor_score:
            mejor_score = score
            mejor_nombre = nombre
            mejor_img = limpio

    return mejor_img, variantes, {"nombre": mejor_nombre, "score": mejor_score}

def calidad_imagen_para_ocr(gray):
    g = asegurar_gris_uint8(gray)
    if g is None or g.size == 0:
        return -9999.0
    zona = recortar_zona_caracteres(g)
    b, _, info = binarizacion_robusta(zona)
    # Laplaciano ayuda, pero no debe dominar porque rayas también aumentan alta frecuencia.
    lap = varianza_laplaciano(zona)
    lap_bonus = min(30.0, np.log1p(max(lap, 0.0)) * 4.0)
    return float(info.get("score", -9999.0)) + lap_bonus

def extraer_strip_desde_banda_caracteres(binary_inv, margen_x=35, margen_y=20, modo_celeste=False):
    if binary_inv is None or binary_inv.size == 0:
        return None

    banda_limpia = limpiar_bordes_componentes_binary_inv(binary_inv.copy(), modo_celeste=modo_celeste)
    ys, xs = np.where(banda_limpia > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    h, w = banda_limpia.shape[:2]
    bx1 = max(0, int(xs.min()) - margen_x)
    bx2 = min(w, int(xs.max()) + margen_x + 1)
    by1 = max(0, int(ys.min()) - margen_y)
    by2 = min(h, int(ys.max()) + margen_y + 1)
    if bx2 <= bx1 or by2 <= by1:
        return None

    strip_inv = banda_limpia[by1:by2, bx1:bx2]
    strip = cv2.bitwise_not(strip_inv)
    strip = cv2.copyMakeBorder(strip, 12, 12, 18, 18, cv2.BORDER_CONSTANT, value=255)
    return strip

def recortar_zona_caracteres(gray):
    if gray is None or gray.size == 0:
        return gray

    h, w = gray.shape[:2]
    y1 = int(h * 0.30)
    y2 = int(h * 0.88)
    x1 = int(w * 0.03)
    x2 = int(w * 0.97)

    if y2 <= y1 or x2 <= x1:
        return gray

    return gray[y1:y2, x1:x2]

def binarizaciones(gray):
    gray = asegurar_gris_uint8(gray)
    suave = cv2.GaussianBlur(gray, (3, 3), 0)

    _, otsu_inv = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, otsu = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    adapt_inv = cv2.adaptiveThreshold(
        suave,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8,
    )
    adapt = cv2.bitwise_not(adapt_inv)

    return otsu_inv, otsu, adapt_inv, adapt

def segmentar_letras_negras_celeste(roi_bgr):
    if roi_bgr is None or roi_bgr.size == 0:
        return None, None, None, None, -9999.0

    if len(roi_bgr.shape) == 2:
        gray = roi_bgr.copy()
        hsv = cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
    else:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    h, w = gray.shape[:2]

    # Zona donde normalmente están los caracteres grandes
    y1 = int(h * 0.12)
    y2 = int(h * 0.96)
    x1 = int(w * 0.01)
    x2 = int(w * 0.99)

    zona_gray = gray[y1:y2, x1:x2]
    zona_hsv = hsv[y1:y2, x1:x2]
    
    # Agregar padding antes de binarizar para evitar recortes en la morfologia
    zona_gray = cv2.copyMakeBorder(zona_gray, 15, 15, 10, 10, cv2.BORDER_REPLICATE)
    zona_hsv = cv2.copyMakeBorder(zona_hsv, 15, 15, 10, 10, cv2.BORDER_REPLICATE)
    
    canal_v = zona_hsv[:, :, 2]
    
    if DEBUG_GUARDAR_IMAGENES:
        os.makedirs("results/debug", exist_ok=True)
        cv2.imwrite("results/debug/zona_celeste_recortada.png", zona_gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    zona_clahe = clahe.apply(zona_gray)

    # Letras negras: texto oscuro sobre fondo celeste claro
    _, otsu_inv = cv2.threshold(
        zona_clahe,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    adapt_inv = cv2.adaptiveThreshold(
        zona_clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9
    )

    # Máscara por oscuridad usando canal V
    dark_mask = cv2.inRange(canal_v, 0, 135)

    candidatos = [otsu_inv, adapt_inv, dark_mask]

    kernel_open = np.ones((2, 2), np.uint8)
    kernel_close = np.ones((2, 2), np.uint8)

    mejores = []

    for binaria in candidatos:
        # Variante suave: sin cierre/apertura o mínimo
        clean_suave = binaria.copy()
        
        # Variante fuerte:
        clean_fuerte = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel_open, iterations=1)
        clean_fuerte = cv2.morphologyEx(clean_fuerte, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        for name_var, clean in [("suave", clean_suave), ("fuerte", clean_fuerte)]:
            # Eliminar componentes pequeños
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, 8)
            filtrada = np.zeros_like(clean)

            zh, zw = clean.shape[:2]

            for label in range(1, num_labels):
                x, y, bw, bh, area = stats[label]

                if area < zh * zw * 0.002:
                    continue

                if bh < zh * 0.25 or bh > zh * 0.95:
                    continue

                ratio = bw / float(bh) if bh else 99

                if not (0.08 <= ratio <= 1.2):
                    continue

                # Evitar bordes largos de la placa
                if bw > zw * 0.35 and bh < zh * 0.35:
                    continue

                filtrada[labels == label] = 255
                
            if DEBUG_GUARDAR_IMAGENES and len(mejores) == 0 and name_var == "fuerte":
                # Guardamos la binaria sin filtrar del primer candidato como debug
                cv2.imwrite("results/debug/binaria_letras_negras_sin_filtrar.png", clean)

            score = evaluar_binarizacion_inv(filtrada)
            mejores.append((score, filtrada, name_var))

    mejores.sort(key=lambda x: x[0], reverse=True)
    
    # Extraer la mejor suave y la mejor fuerte
    mejor_fuerte = None
    mejor_suave = None
    mejor_score = -9999.0
    
    for m in mejores:
        if m[2] == "fuerte" and mejor_fuerte is None:
            mejor_fuerte = m[1]
            if mejor_score == -9999.0: mejor_score = m[0]
        if m[2] == "suave" and mejor_suave is None:
            mejor_suave = m[1]
            if mejor_score == -9999.0: mejor_score = m[0]

        # Eliminar componentes pequeños
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, 8)
        filtrada = np.zeros_like(clean)

        zh, zw = clean.shape[:2]

        for label in range(1, num_labels):
            x, y, bw, bh, area = stats[label]

            if area < zh * zw * 0.002:
                continue

            if bh < zh * 0.25 or bh > zh * 0.95:
                continue

            ratio = bw / float(bh) if bh else 99

            if not (0.08 <= ratio <= 1.2):
                continue

            # Evitar bordes largos de la placa
            if bw > zw * 0.35 and bh < zh * 0.35:
                continue

            filtrada[labels == label] = 255
            
        if DEBUG_GUARDAR_IMAGENES:
            # Guardamos la binaria sin filtrar del primer candidato como debug
            if len(mejores) == 0:
                cv2.imwrite("results/debug/binaria_letras_negras_sin_filtrar.png", clean)

    if DEBUG_GUARDAR_IMAGENES:
        if mejor_fuerte is not None:
            cv2.imwrite("results/debug/binaria_letras_negras_filtrada.png", mejor_fuerte)
            cv2.imwrite("results/debug/binarizada_letras_negras.png", mejor_fuerte)

    # Segmentación visual
    contornos, _ = cv2.findContours(
        mejor_fuerte if mejor_fuerte is not None else mejor_suave,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    img_segmentacion = cv2.cvtColor(zona_gray, cv2.COLOR_GRAY2BGR)

    for cnt in contornos:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh >= (mejor_fuerte.shape[0] if mejor_fuerte is not None else mejor_suave.shape[0]) * 0.25:
            cv2.rectangle(img_segmentacion, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    strip_fuerte = extraer_strip_desde_banda_caracteres(mejor_fuerte, modo_celeste=True) if mejor_fuerte is not None else None
    strip_suave = extraer_strip_desde_banda_caracteres(mejor_suave, modo_celeste=True) if mejor_suave is not None else None

    if DEBUG_GUARDAR_IMAGENES:
        if strip_fuerte is not None:
            cv2.imwrite("results/debug/strip_caracteres.png", strip_fuerte)
            cv2.imwrite("results/debug/strip_fuerte.png", strip_fuerte)
        if strip_suave is not None:
            cv2.imwrite("results/debug/strip_suave.png", strip_suave)

    return mejor_fuerte, mejor_suave, img_segmentacion, strip_fuerte, strip_suave, mejor_score, zona_gray, zona_clahe, canal_v

def limpiar_bordes_componentes_binary_inv(binary_inv, modo_celeste=False):
    if binary_inv is None or binary_inv.size == 0:
        return binary_inv

    img = binary_inv.copy()
    h, w = img.shape[:2]
    if h < 5 or w < 5:
        return img

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((img > 0).astype(np.uint8), 8)

    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]

        toca_izq = x <= 2
        toca_der = (x + bw) >= (w - 2)
        toca_arriba = y <= 2
        toca_abajo = (y + bh) >= (h - 2)

        es_linea_vertical_borde = (toca_izq or toca_der) and bh > h * 0.55 and bw < w * 0.12
        es_linea_horizontal_borde = (toca_arriba or toca_abajo) and bw > w * 0.55 and bh < h * 0.18

        ratio = bw / float(bh) if bh > 0 else 99
        es_caracter_valido = area > h * w * 0.005 and 0.15 <= ratio <= 1.2 and bh >= h * 0.35

        # En modo celeste, no borramos el componente si toca los bordes
        # siempre y cuando tenga aspecto de caracter válido y NO sea una línea de marco evidente.
        if modo_celeste and es_caracter_valido and not (es_linea_vertical_borde or es_linea_horizontal_borde):
            continue

        # Ruido grande del marco, no caracteres.
        if es_linea_vertical_borde or es_linea_horizontal_borde or (toca_izq and not es_caracter_valido) or (toca_der and not es_caracter_valido) or (toca_arriba and not es_caracter_valido) or (toca_abajo and not es_caracter_valido):
            img[labels == label] = 0

    # Segunda pasada: eliminar columnas/filas extremas con demasiada tinta.
    # Esto ataca bordes que quedaron conectados con suciedad.
    for _ in range(2):
        h, w = img.shape[:2]
        if w <= 10 or h <= 10:
            break

        col_black_ratio = np.mean(img > 0, axis=0)
        row_black_ratio = np.mean(img > 0, axis=1)

        left = 0
        while left < w - 1 and col_black_ratio[left] > 0.55:
            left += 1

        right = w - 1
        while right > 0 and col_black_ratio[right] > 0.55:
            right -= 1

        top = 0
        while top < h - 1 and row_black_ratio[top] > 0.55:
            top += 1

        bottom = h - 1
        while bottom > 0 and row_black_ratio[bottom] > 0.55:
            bottom -= 1

        if left > 0:
            img[:, :left + 1] = 0
        if right < w - 1:
            img[:, right:] = 0
        if top > 0:
            img[:top + 1, :] = 0
        if bottom < h - 1:
            img[bottom:, :] = 0

    return img

def limpiar_strip_para_ocr(strip, remover_bordes_laterales=False):
    gray = asegurar_gris_uint8(strip)
    if gray is None or gray.size == 0:
        return strip

    # Normalizar a texto negro sobre blanco.
    if np.mean(gray) < 127:
        gray = cv2.bitwise_not(gray)

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Foreground negro.
    fg = bw < 128
    if not np.any(fg):
        return gray

    h, w = bw.shape[:2]

    # Eliminar componentes de borde en foreground negro.
    inv = np.where(fg, 255, 0).astype(np.uint8)
    inv = limpiar_bordes_componentes_binary_inv(inv)
    
    if remover_bordes_laterales:
        # Encontrar componentes conectados y remover los que tocan los extremos laterales
        # O que son ruido vertical (muy delgados y pegados a los bordes).
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inv, connectivity=8)
        
        margen_lateral = int(w * 0.12)
        
        for i in range(1, num_labels):
            x, y, w_comp, h_comp, area = stats[i]
            
            toca_borde_izq = (x <= 2)
            toca_borde_der = ((x + w_comp) >= w - 3)
            
            esta_en_margen = (x < margen_lateral) or ((x + w_comp) > w - margen_lateral)
            es_ruido_vertical = (h_comp > h * 0.5) and (w_comp < w * 0.04) # muy delgado
            
            # No borrar si tiene proporción/tamaño de letra válida (ej. P, M, B, K)
            es_letra_valida = (w_comp > w * 0.05) and (0.2 < (w_comp / float(h_comp)) < 1.5)
            
            if (toca_borde_izq or toca_borde_der) or (esta_en_margen and es_ruido_vertical and not es_letra_valida):
                inv[labels == i] = 0

    fg = inv > 0

    if not np.any(fg):
        return gray

    ys, xs = np.where(fg)
    x1 = max(0, int(xs.min()) - 10)
    x2 = min(w, int(xs.max()) + 11)
    y1 = max(0, int(ys.min()) - 8)
    y2 = min(h, int(ys.max()) + 9)

    clean = bw[y1:y2, x1:x2]

    # Adelgazar texto muy grueso: en imagen binaria con texto negro/fondo blanco,
    # dilatar el blanco reduce un poco las manchas negras.
    kernel = np.ones((2, 2), np.uint8)
    clean_thin = cv2.dilate(clean, kernel, iterations=1)

    clean_thin = cv2.copyMakeBorder(clean_thin, 18, 18, 26, 26, cv2.BORDER_CONSTANT, value=255)
    return clean_thin

def extraer_strip_global_caracteres(binary_morf, margen=8):
    if binary_morf is None or binary_morf.size == 0:
        return None

    h, w = binary_morf.shape[:2]

    # Banda donde están los caracteres grandes. Evita PERU arriba y pernos abajo.
    y1 = int(h * 0.30)
    y2 = int(h * 0.88)
    x1 = int(w * 0.02)
    x2 = int(w * 0.98)

    banda = binary_morf[y1:y2, x1:x2]
    if banda.size == 0:
        return None

    # Limpiar ruido fino sin romper trazos grandes.
    kernel = np.ones((2, 2), np.uint8)
    banda_limpia = cv2.morphologyEx(banda, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Quitar líneas del marco de la placa antes de calcular el bbox.
    # Esto evita que el borde izquierdo se lea como un dígito 1.
    banda_limpia = limpiar_bordes_componentes_binary_inv(banda_limpia)

    # En binary_morf el texto queda blanco sobre fondo negro.
    ys, xs = np.where(banda_limpia > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    bx1 = max(0, int(xs.min()) - margen)
    bx2 = min(banda_limpia.shape[1], int(xs.max()) + margen)
    by1 = max(0, int(ys.min()) - margen)
    by2 = min(banda_limpia.shape[0], int(ys.max()) + margen)

    if bx2 <= bx1 or by2 <= by1:
        return None

    strip_inv = banda_limpia[by1:by2, bx1:bx2]

    # Convertimos a texto negro sobre fondo blanco para GUI y OCR.
    strip = cv2.bitwise_not(strip_inv)

    # Borde blanco para que OCR no corte las letras de los extremos.
    strip = cv2.copyMakeBorder(strip, 12, 12, 18, 18, cv2.BORDER_CONSTANT, value=255)
    return strip

def segmentar_y_leer(img_restaurada, img_clahe, reader=None, img_bgr=None, is_celeste=False):
    base = asegurar_gris_uint8(img_restaurada)
    zona_base = recortar_zona_caracteres(base)
    if zona_base is None or zona_base.size == 0:
        zona_base = base

    # Aumentar un poco la banda antes de binarizar mejora trazos para OCR.
    if zona_base.shape[0] < 90:
        escala = 90.0 / max(1.0, float(zona_base.shape[0]))
        zona_base = cv2.resize(
            zona_base,
            (max(1, int(zona_base.shape[1] * escala)), 90),
            interpolation=cv2.INTER_CUBIC,
        )

    binary_morf, bins, info_bin = binarizacion_robusta(zona_base)
    print(f"[DEBUG BIN] Mejor binarización estandar: {info_bin.get('nombre')} | score={info_bin.get('score'):.1f}")

    contornos, _ = cv2.findContours(binary_morf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contornos = cv2.cvtColor(zona_base, cv2.COLOR_GRAY2BGR)

    altura_img, ancho_img = binary_morf.shape[:2]
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / float(h) if h > 0 else 999
        area = w * h
        if h < altura_img * 0.18:
            continue
        if area < (altura_img * ancho_img) * 0.001:
            continue
        if not (0.08 <= ratio <= 4.5):
            continue
        cv2.rectangle(img_contornos, (x, y), (x + w, y + h), (0, 255, 0), 2)

    strip_visual = extraer_strip_desde_banda_caracteres(binary_morf)
    imagenes_caracteres = []
    img_strip = None
    if strip_visual is not None:
        imagenes_caracteres = [strip_visual]
        img_strip = strip_visual.copy()

    # Asegurar claves esperadas por procesar_fragmento.
    if "otsu_inv" not in bins:
        bins["otsu_inv"] = binary_morf
    if "otsu" not in bins:
        bins["otsu"] = cv2.bitwise_not(binary_morf)
    if "adapt_inv" not in bins:
        bins["adapt_inv"] = binary_morf
    if "adapt" not in bins:
        bins["adapt"] = cv2.bitwise_not(binary_morf)
    bins["zona_base"] = zona_base

    # Variante de letras negras celeste
    if img_bgr is not None:
        bin_negras_fuerte, bin_negras_suave, img_seg_celeste, strip_fuerte, strip_suave, score_celeste, z_gray, z_clahe, c_v = segmentar_letras_negras_celeste(img_bgr)
        if bin_negras_fuerte is not None:
            # Añadir como variantes siempre
            bins["letras_negras_inv"] = bin_negras_fuerte
            bins["letras_negras"] = cv2.bitwise_not(bin_negras_fuerte)
            if strip_fuerte is not None:
                bins["strip_letras_negras"] = strip_fuerte
            if strip_suave is not None:
                bins["strip_letras_negras_suave"] = strip_suave
                
            bins["zona_gray_celeste"] = z_gray
            bins["zona_clahe_celeste"] = z_clahe
            bins["canal_v_celeste"] = c_v
                
            print(f"[DEBUG BIN] Binarización celeste | score={score_celeste:.1f}")
            
            # Si estamos en modo celeste y el score es decente, reemplazar las visualizaciones principales
            if is_celeste and score_celeste > -5000:  
                binary_morf = bin_negras_fuerte
                img_contornos = img_seg_celeste
                img_strip = strip_fuerte
                if strip_fuerte is not None:
                    imagenes_caracteres = [strip_fuerte]

    return binary_morf, img_contornos, imagenes_caracteres, img_strip, bins

