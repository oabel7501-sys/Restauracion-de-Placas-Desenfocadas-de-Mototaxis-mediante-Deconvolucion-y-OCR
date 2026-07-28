import cv2
import json
import os
import pandas as pd
from utils.config import DEBUG_GUARDAR_IMAGENES

def guardar_debug_pipeline(etapas, ruta_base="results/debug"):
    """
    Guarda las imágenes intermitentes y reportes de texto/json para debugging.
    'etapas' es el diccionario que devuelve procesar_fragmento.
    """
    if not DEBUG_GUARDAR_IMAGENES:
        return
        
    os.makedirs(ruta_base, exist_ok=True)
    
    if "img_roi" in etapas and etapas["img_roi"] is not None:
        cv2.imwrite(os.path.join(ruta_base, "roi.png"), etapas["img_roi"])
        
    if "img_restaurada" in etapas and etapas["img_restaurada"] is not None:
        cv2.imwrite(os.path.join(ruta_base, "preprocesada.png"), etapas["img_restaurada"])
        
    if "img_binarizada" in etapas and etapas["img_binarizada"] is not None:
        cv2.imwrite(os.path.join(ruta_base, "binarizada.png"), etapas["img_binarizada"])
        
    if "img_contornos" in etapas and etapas["img_contornos"] is not None:
        cv2.imwrite(os.path.join(ruta_base, "segmentacion.png"), etapas["img_contornos"])
        
    if "img_strip" in etapas and etapas["img_strip"] is not None:
        cv2.imwrite(os.path.join(ruta_base, "strip.png"), etapas["img_strip"])
        
    if "lecturas_brutas" in etapas:
        with open(os.path.join(ruta_base, "ocr_raw.txt"), "w", encoding="utf-8") as f:
            for l in etapas["lecturas_brutas"]:
                f.write(str(l) + "\n")
                
    if "candidatos" in etapas and etapas["candidatos"]:
        df = pd.DataFrame(etapas["candidatos"])
        df.to_csv(os.path.join(ruta_base, "candidatos.csv"), index=False)
        
    # Guardar un json limpio (excluyendo matrices numpy)
    resultado = {
        "placa": etapas.get("placa"),
        "valido": etapas.get("valido"),
        "tipo": etapas.get("tipo"),
        "score": etapas.get("score")
    }
    with open(os.path.join(ruta_base, "resultado.json"), "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=4)
