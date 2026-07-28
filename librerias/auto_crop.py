import cv2
import numpy as np
import os
from utils.config import (
    HSV_LOWER_CELESTE, HSV_UPPER_CELESTE, DEBUG_GUARDAR_IMAGENES, AUTO_CROP_MARGIN,
    MIN_PLATE_AREA_RATIO, MAX_PLATE_AREA_RATIO, PLATE_ASPECT_MIN, PLATE_ASPECT_MAX,
    MIN_TEXT_COMPONENTS_AUTO, MIN_TEXT_COMPONENTS_DUDOSO,
    MIN_DARK_RATIO_AUTO, MAX_DARK_RATIO_AUTO
)
from utils.roi import order_points
from utils.segmentation import segmentar_letras_negras_celeste

def evaluar_candidato(cnt, img_bgr):
    """
    Evalúa la calidad del contorno candidato en base a área relativa, rectangularidad y proporción.
    """
    area = cv2.contourArea(cnt)
    total_area = img_bgr.shape[0] * img_bgr.shape[1]
    area_ratio = area / total_area if total_area > 0 else 0
    
    if not (MIN_PLATE_AREA_RATIO <= area_ratio <= MAX_PLATE_AREA_RATIO):
        return 0.0, area_ratio, 0.0, 0.0
        
    rect = cv2.minAreaRect(cnt)
    width, height = rect[1][0], rect[1][1]
    if width == 0 or height == 0:
        return 0.0, area_ratio, 0.0, 0.0
        
    aspect_ratio = float(width) / height
    if aspect_ratio < 1.0:
        aspect_ratio = 1.0 / aspect_ratio
        
    if not (PLATE_ASPECT_MIN <= aspect_ratio <= PLATE_ASPECT_MAX):
        return 0.0, area_ratio, aspect_ratio, 0.0
        
    box_area = width * height
    rectangularity = area / box_area if box_area > 0 else 0
    if rectangularity < 0.4:
        return 0.0, area_ratio, aspect_ratio, rectangularity
        
    # Puntaje base
    score = area * rectangularity
    return score, area_ratio, aspect_ratio, rectangularity

def extraer_cuadrilatero_refinado(img_bgr, rect):
    """
    Refinamiento en dos etapas: recorta la zona macro y usa Canny/Morfología para hallar las 4 esquinas.
    Devuelve los 4 puntos reales en la imagen original, o None si falla.
    """
    box = cv2.boxPoints(rect)
    pts = np.asarray(box, dtype=np.int32)
    
    # Expandir ROI macro para incluir todo el borde posible
    margen_macro = 20
    x_coords = [p[0] for p in pts]
    y_coords = [p[1] for p in pts]
    x_min = max(0, int(min(x_coords)) - margen_macro)
    y_min = max(0, int(min(y_coords)) - margen_macro)
    x_max = min(img_bgr.shape[1], int(max(x_coords)) + margen_macro)
    y_max = min(img_bgr.shape[0], int(max(y_coords)) + margen_macro)
    
    if x_max <= x_min or y_max <= y_min:
        return pts, "minAreaRect fallback"
        
    roi_macro = img_bgr[y_min:y_max, x_min:x_max]
    
    # Micro refinamiento
    gray = cv2.cvtColor(roi_macro, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    cnts, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mejor_poli = None
    mejor_area_poli = 0
    
    for c in cnts:
        area = cv2.contourArea(c)
        if area > roi_macro.shape[0] * roi_macro.shape[1] * 0.15:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            
            if len(approx) == 4 and cv2.isContourConvex(approx):
                if area > mejor_area_poli:
                    mejor_area_poli = area
                    mejor_poli = approx
                    
    if mejor_poli is not None:
        # Convertir a coordenadas globales
        pts_globales = []
        for p in mejor_poli:
            pts_globales.append([p[0][0] + x_min, p[0][1] + y_min])
        return np.array(pts_globales, dtype="float32"), "approxPolyDP"
        
    return pts, "minAreaRect fallback"

def contar_componentes_texto(bin_negras_fuerte):
    """
    Cuenta cuántos componentes parecidos a caracteres hay en la binarización.
    """
    if bin_negras_fuerte is None:
        return 0
        
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_negras_fuerte, 8)
    zh, zw = bin_negras_fuerte.shape[:2]
    
    valid_count = 0
    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        
        if area < zh * zw * 0.002:
            continue
        if bh < zh * 0.25 or bh > zh * 0.95:
            continue
        ratio = bw / float(bh) if bh else 99
        if not (0.08 <= ratio <= 1.2):
            continue
        if bw > zw * 0.35 and bh < zh * 0.35:
            continue
            
        valid_count += 1
        
    return valid_count

def detectar_placa_celeste(img_bgr):
    """
    Intenta detectar una placa celeste en la imagen usando HSV.
    Evalúa múltiples candidatos y refina la perspectiva.
    Devuelve (recorte_normalizado, puntos_roi, mensaje_estado)
    """
    if img_bgr is None:
        return None, None, "No se detectó placa celeste. Use recorte manual."

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(HSV_LOWER_CELESTE, dtype="uint8")
    upper = np.array(HSV_UPPER_CELESTE, dtype="uint8")

    # Crear máscara
    mask = cv2.inRange(hsv, lower, upper)

    # Morfología matemática
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=1)

    if DEBUG_GUARDAR_IMAGENES:
        os.makedirs("results/debug", exist_ok=True)
        cv2.imwrite("results/debug/mascara_celeste.png", mask_opened)
        cv2.imwrite("results/debug/imagen_original.png", img_bgr)

    # Encontrar contornos
    contours, _ = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_contours = img_bgr.copy()
    
    candidatos = []
    for cnt in contours:
        score, area_ratio, aspect_ratio, rectangularity = evaluar_candidato(cnt, img_bgr)
        if score > 0:
            candidatos.append((score, cnt, area_ratio, aspect_ratio, rectangularity))

    if not candidatos:
        return None, None, "No se detectó placa celeste. Use recorte manual."

    candidatos.sort(key=lambda x: x[0], reverse=True)
    
    if DEBUG_GUARDAR_IMAGENES:
        for i, (score, cnt, area_ratio, aspect_ratio, rectangularity) in enumerate(candidatos[:3]):
            color = (0, 255, 0) if i == 0 else (0, 165, 255)
            cv2.drawContours(img_contours, [cnt], -1, color, 2)
            cv2.putText(img_contours, f"#{i+1} s:{score:.0f}", tuple(cnt[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imwrite("results/debug/contornos_candidatos.png", img_contours)

    # Preparar archivo de debug de scores
    debug_log = ""
    if DEBUG_GUARDAR_IMAGENES:
        debug_log += f"Total candidatos validos iniciales: {len(candidatos)}\n\n"

    mejor_recorte = None
    mejor_pts = None
    max_score_final = -1
    mejor_es_dudoso = False
    
    for i, (score_geom, cnt, area_ratio, aspect_ratio, rectangularity) in enumerate(candidatos[:5]):
        rect = cv2.minAreaRect(cnt)
        pts_reales, metodo = extraer_cuadrilatero_refinado(img_bgr, rect)
        
        if pts_reales is None:
            continue
            
        pts = order_points(pts_reales)
        
        # Aplicar margen configurable
        pts[0][0] -= AUTO_CROP_MARGIN; pts[0][1] -= AUTO_CROP_MARGIN
        pts[1][0] += AUTO_CROP_MARGIN; pts[1][1] -= AUTO_CROP_MARGIN
        pts[2][0] += AUTO_CROP_MARGIN; pts[2][1] += AUTO_CROP_MARGIN
        pts[3][0] -= AUTO_CROP_MARGIN; pts[3][1] += AUTO_CROP_MARGIN
        
        # Calcular ancho y alto
        (tl, tr, br, bl) = pts
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        if maxWidth == 0 or maxHeight == 0:
            continue
            
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
            
        M = cv2.getPerspectiveTransform(pts, dst)
        recorte_warp = cv2.warpPerspective(img_bgr, M, (maxWidth, maxHeight))
        
        # Normalizar
        escala = min(360.0 / float(maxWidth), 150.0 / float(maxHeight))
        nuevo_w = max(1, int(maxWidth * escala))
        nuevo_h = max(1, int(maxHeight * escala))
        
        recorte_resize = cv2.resize(recorte_warp, (nuevo_w, nuevo_h), interpolation=cv2.INTER_CUBIC)
        
        top = (150 - nuevo_h) // 2
        bottom = 150 - nuevo_h - top
        left = (360 - nuevo_w) // 2
        right = 360 - nuevo_w - left
        
        recorte_normalizado = cv2.copyMakeBorder(recorte_resize, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        if DEBUG_GUARDAR_IMAGENES:
            cv2.imwrite(f"results/debug/roi_candidato_{i+1}.png", recorte_normalizado)
        
        # Validar letras negras
        resultado = segmentar_letras_negras_celeste(recorte_normalizado)
        bin_negras_fuerte = resultado[0]
        bin_negras_suave = resultado[1]
        score_celeste = resultado[5]
        
        gray_recorte = cv2.cvtColor(recorte_normalizado, cv2.COLOR_BGR2GRAY)
        dark_pixels = np.sum(gray_recorte < 100)
        total_pixels = gray_recorte.size
        dark_ratio = dark_pixels / total_pixels if total_pixels > 0 else 0
        
        comp_fuerte = contar_componentes_texto(bin_negras_fuerte)
        comp_suave = contar_componentes_texto(bin_negras_suave)
        cantidad_componentes = max(comp_fuerte, comp_suave)
        
        # Integrar texto y geometria
        bonus_texto = 1.0 + (cantidad_componentes * 0.25)
        score_final = score_geom * bonus_texto
        
        motivo = "Ninguno"
        if dark_ratio < MIN_DARK_RATIO_AUTO or dark_ratio > MAX_DARK_RATIO_AUTO:
            score_final *= 0.1
            motivo = f"Penalizado por dark_ratio ({dark_ratio:.3f})"
            
        dudoso = (cantidad_componentes < MIN_TEXT_COMPONENTS_AUTO)
        if dudoso and motivo == "Ninguno":
            motivo = f"Pocos componentes de texto ({cantidad_componentes})"
            
        debug_log += f"--- Candidato {i+1} ---\n"
        debug_log += f"Area Ratio: {area_ratio:.5f}\n"
        debug_log += f"Aspect Ratio: {aspect_ratio:.3f}\n"
        debug_log += f"Rectangularidad: {rectangularity:.3f}\n"
        debug_log += f"Dark Ratio: {dark_ratio:.4f}\n"
        debug_log += f"Componentes Texto: {cantidad_componentes} (F:{comp_fuerte}, S:{comp_suave})\n"
        debug_log += f"Score Geom: {score_geom:.2f}\n"
        debug_log += f"Score Final ROI: {score_final:.2f}\n"
        debug_log += f"Es Dudoso: {dudoso}\n"
        debug_log += f"Motivo/Nota: {motivo}\n\n"
        
        if score_final > max_score_final:
            max_score_final = score_final
            mejor_recorte = recorte_normalizado
            mejor_pts = pts
            mejor_es_dudoso = dudoso
            
    if DEBUG_GUARDAR_IMAGENES:
        try:
            with open("results/debug/roi_auto_score.txt", "w") as f:
                f.write(debug_log)
        except: pass
        
    if mejor_recorte is not None:
        if DEBUG_GUARDAR_IMAGENES:
            cv2.imwrite("results/debug/roi_auto_final.png", mejor_recorte)
            
        if mejor_es_dudoso:
            return mejor_recorte, mejor_pts.tolist(), "Recorte automático dudoso."
        else:
            return mejor_recorte, mejor_pts.tolist(), "OK"
            
    return None, None, "No se detectó placa válida. Use recorte manual."
