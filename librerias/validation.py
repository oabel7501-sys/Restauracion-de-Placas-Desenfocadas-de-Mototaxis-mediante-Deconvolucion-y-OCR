import re
import itertools

FORMATOS_PLACA = [
    ("MOTO INVERTIDO", "DDDDDL", lambda s: f"{s[:4]}-{s[4:]}", 45),

    ("MOTO INVERTIDO", "DDDDLL", lambda s: f"{s[:4]}-{s[4:]}", 34),

    # Placas antiguas mixtas como P7-2478 / P6-8150: letra + dígito + 4 dígitos.
    ("MOTO ANTIGUA MIXTA", "LDDDDD", lambda s: f"{s[:2]}-{s[2:]}", 30),

    ("MOTO NORMAL", "LLDDDD", lambda s: f"{s[:2]}-{s[2:]}", 18),
    ("AUTO LIVIANO", "LLLDDD", lambda s: f"{s[:3]}-{s[3:]}", 14),
    ("AUTO NUEVO", "LDLDDD", lambda s: f"{s[:3]}-{s[3:]}", 14),
    ("AUTO ANTIGUO", "LLLDDDD", lambda s: f"{s[:3]}-{s[3:]}", 6),
    ("MOTO ANTIGUA", "LLDDDDD", lambda s: f"{s[:2]}-{s[2:]}", 8),
]

LETRA_A_NUM = {
    "O": ["0"],
    "Q": ["0"],
    "D": ["0"],
    "C": ["0"],
    "U": ["0"],
    "I": ["1"],
    "L": ["1"],
    "Z": ["2"],
    "S": ["5", "3"],
    "B": ["8", "3"],
    "G": ["6"],
    "A": ["4"],
    "T": ["7"],
    # En placas borrosas, 9 puede verse como P/R y viceversa.
    "P": ["9"],
    "R": ["9"],
}

NUM_A_LETRA = {
    "0": ["O", "D", "Q", "C"],
    "1": ["I", "L"],
    "2": ["Z"],
    "3": ["B", "S"],
    "4": ["A"],
    "5": ["S"],
    "6": ["G"],
    "7": ["T"],
    "8": ["B"],
    "9": ["P", "R", "B"],  # B solo se favorecerá con reglas contextuales en validación.
}

LETRAS_SIMILARES = {
    "P": ["P", "R", "D", "H"],
    "R": ["R", "P"],
    "D": ["D", "P", "O"],
    "O": ["O", "D", "Q", "C"],
    "Q": ["Q", "O"],
    "C": ["C", "O", "G"],
    "I": ["I", "L", "C"],
    "L": ["L", "I"],
    "E": ["E", "F"],
    "F": ["F", "E", "G"],
    "U": ["U", "O"],
    "B": ["B", "R"],
    "S": ["S"],
    "A": ["A"],
    "G": ["G", "F"],
    "T": ["T"],
    "H": ["H", "P", "M"],
    "M": ["M", "H"],
}

DIGITOS_SIMILARES = {
    "3": ["5"],
    "5": ["3", "8"],
    "8": ["5"],
    "4": ["6"],
    "6": ["4"],
}

BASURA_PLACA = [
    "PERU", "PERO", "PRU", "PEPU", "REPUBLICA", "DEL", "PLACA", "RODAJE"
]

def limpiar_lectura_ocr(texto):
    texto = str(texto).upper()
    texto = re.sub(r"[^A-Z0-9]", "", texto)

    for basura in BASURA_PLACA:
        texto = texto.replace(basura, "")

    # Algunos OCR devuelven PE antes del número grande.
    if texto.startswith("PE") and len(texto) > 6:
        texto = texto[2:]

    return texto

def extraer_fragmentos(texto):
    texto = limpiar_lectura_ocr(texto)
    fragmentos = []
    for length in (6, 7):
        if len(texto) >= length:
            for i in range(len(texto) - length + 1):
                fragmentos.append(texto[i:i + length])

    if len(texto) in (6, 7):
        fragmentos.insert(0, texto)

    return list(dict.fromkeys(fragmentos))

def opciones_para_char(char_ocr, esperado):
    opciones = {}

    if esperado == "D":
        if char_ocr.isdigit():
            # Mantener el dígito leído tiene mucha prioridad.
            opciones[char_ocr] = max(opciones.get(char_ocr, 0), 18)
            # Las sustituciones entre dígitos quedan casi desactivadas por seguridad.
            for n in DIGITOS_SIMILARES.get(char_ocr, []):
                opciones[n] = max(opciones.get(n, 0), 2)

        if char_ocr in LETRA_A_NUM:
            for n in LETRA_A_NUM[char_ocr]:
                opciones[n] = max(opciones.get(n, 0), 8)

    elif esperado == "L":
        if char_ocr.isalpha():
            for letra in LETRAS_SIMILARES.get(char_ocr, [char_ocr]):
                if letra == char_ocr:
                    opciones[letra] = max(opciones.get(letra, 0), 18)
                else:
                    opciones[letra] = max(opciones.get(letra, 0), 8)

        if char_ocr in NUM_A_LETRA:
            for letra in NUM_A_LETRA[char_ocr]:
                opciones[letra] = max(opciones.get(letra, 0), 7)

    return list(opciones.items())

def generar_candidatos_para_formato(fragmento, mascara):
    if len(fragmento) != len(mascara):
        return []

    opciones_posicion = []
    for char_ocr, esperado in zip(fragmento, mascara):
        opciones = opciones_para_char(char_ocr, esperado)
        if not opciones:
            return []
        opciones_posicion.append(opciones)

    candidatos = []
    for combinacion in itertools.product(*opciones_posicion):
        texto = "".join(item[0] for item in combinacion)
        score_base = sum(item[1] for item in combinacion)
        candidatos.append((texto, score_base))

    return candidatos

def contar_mutaciones(candidato, fragmento):
    return sum(1 for a, b in zip(candidato, fragmento) if a != b)

def peso_fuente_ocr(nombre):
    nombre = str(nombre or "").lower()

    # OCR clásico sobre zona de caracteres: fuente más confiable para placas ya legibles.
    if "tess_zona_lowlight" in nombre or "tess_zona_clahe_fuerte" in nombre:
        return 82
    if "tess_zona_gray" in nombre or "tess_zona_clahe" in nombre or "tess_zona_sharp" in nombre:
        return 78
    if "tess_zona" in nombre:
        return 68

    # Zonas recortadas: más confiables que la imagen completa.
    if "zona_lowlight" in nombre or "zona_clahe_fuerte" in nombre:
        return 74
    if "zona_sharp" in nombre:
        return 70
    if "zona_gray" in nombre or "zona_clahe" in nombre:
        return 68
    if "zona_restaurada" in nombre:
        return 55
    if "zona_otsu" in nombre or "zona_adapt" in nombre:
        return 42

    # El strip/panel 5 es útil, pero no debe imponer una lectura falsa si salió contaminado.
    if "strip" in nombre:
        return 50

    # Imagen completa: baja prioridad porque PERU, tornillos y bordes contaminan.
    if "completa" in nombre:
        return 8

    return 15

def normalizar_lectura_item(item):
    if isinstance(item, dict):
        texto = limpiar_lectura_ocr(item.get("texto", ""))
        fuente = item.get("fuente", "desconocida")
        conf = float(item.get("conf", 1.0) or 0.0)
        peso = float(item.get("peso", peso_fuente_ocr(fuente)))
    else:
        texto = limpiar_lectura_ocr(item)
        fuente = "legacy"
        conf = 1.0
        peso = 15.0

    return texto, fuente, conf, peso

def evaluar_fallback_numerico_5(lecturas_info):
    numericos = []
    for lectura_limpia, info in lecturas_info.items():
        if not (len(lectura_limpia) == 5 and lectura_limpia.isdigit()):
            continue

        fuentes_join = ",".join(sorted(info["fuentes"])).lower()
        if "completa" in fuentes_join or "combinado" in fuentes_join:
            continue

        score = 70.0
        score += float(info.get("peso_max", 0.0))
        score += min(int(info.get("count", 0)), 6) * 14.0
        score += min(max(float(info.get("conf_max", 0.0)), 0.0), 1.0) * 10.0

        if info.get("tiene_strip"):
            score += 18.0
        if "zona_gray" in fuentes_join or "zona_clahe" in fuentes_join or "zona_sharp" in fuentes_join:
            score += 20.0
        if "tess" in fuentes_join:
            score += 15.0

        # Bono masivo para igualar la prioridad de lecturas exactas en otros formatos
        score += 200.0

        numericos.append({
            "score": score,
            "placa": lectura_limpia,
            "tipo": "MOTO NUMERICA",
            "candidato": lectura_limpia,
            "fragmento": lectura_limpia,
            "lectura": lectura_limpia,
            "mutaciones": 0,
            "fuentes": ",".join(sorted(info["fuentes"])),
            "count": info["count"],
            "tiene_strip": info["tiene_strip"],
        })

    numericos.sort(key=lambda x: (x["score"], x["count"], 1 if x["tiene_strip"] else 0), reverse=True)
    return numericos

def validar_pool_lecturas(lecturas, is_celeste=False):
    evaluados = []

    # Consolidar por texto, conservando cuántas veces apareció y su mejor fuente.
    lecturas_info = {}
    for item in lecturas:
        limpia, fuente, conf, peso = normalizar_lectura_item(item)
        if len(limpia) < 2:
            continue

        if limpia not in lecturas_info:
            lecturas_info[limpia] = {
                "lectura": limpia,
                "fuentes": set(),
                "count": 0,
                "peso_max": 0.0,
                "conf_max": 0.0,
                "tiene_strip": False,
            }

        info = lecturas_info[limpia]
        info["fuentes"].add(str(fuente))
        info["count"] += 1
        info["peso_max"] = max(info["peso_max"], peso)
        info["conf_max"] = max(info["conf_max"], conf)
        if "strip" in str(fuente).lower():
            info["tiene_strip"] = True

    numericos_fallback = evaluar_fallback_numerico_5(lecturas_info)

    for lectura_limpia, info in lecturas_info.items():
        if len(lectura_limpia) < 5:
            continue

        fragmentos = extraer_fragmentos(lectura_limpia)

        for frag in fragmentos:
            for tipo, mascara, formatear, bonus_formato in FORMATOS_PLACA:
                if len(frag) != len(mascara):
                    continue
                if mascara == "DDDDDL":
                    if len(frag) < 5 or frag[4] not in ["9", "P", "R"]:
                        continue

                if mascara == "LDDDDD":
                    if len(frag) != 6:
                        continue
                    if frag[0].isdigit() and frag[0] not in NUM_A_LETRA:
                        continue
                    if not frag[1:].isdigit():
                        # Permitimos solo confusiones letra->número en posiciones numéricas,
                        # no cadenas muy contaminadas.
                        parecen_num = sum(1 for c in frag[1:] if c.isdigit() or c in LETRA_A_NUM)
                        if parecen_num < 5:
                            continue

                for cand, score_base in generar_candidatos_para_formato(frag, mascara):
                    score = score_base + bonus_formato
                    mutaciones = contar_mutaciones(cand, frag)

                    # Prioridad por fuente OCR.
                    score += info["peso_max"]
                    score += min(info["count"], 5) * 7
                    score += min(max(info["conf_max"], 0.0), 1.0) * 8

                    # Premiar lecturas exactas; castigar mutaciones fuertes.
                    if cand == frag:
                        score += 25
                    else:
                        score -= mutaciones * 7

                    # Fragmento extraído de una lectura larga es menos confiable.
                    if len(lectura_limpia) > len(frag):
                        score -= 12

                    if info["tiene_strip"]:
                        score += 25

                    # Para formato 9999AA, los primeros 4 deben parecer números.
                    primeros_4_parecen_num = sum(
                        1 for c in frag[:4]
                        if c.isdigit() or c in LETRA_A_NUM
                    )
                    if mascara == "DDDDLL" and primeros_4_parecen_num >= 3:
                        score += 25

                    primeros_5_parecen_num = sum(
                        1 for c in frag[:5]
                        if c.isdigit() or c in LETRA_A_NUM
                    )
                    if mascara == "DDDDDL":
                        if primeros_5_parecen_num >= 4:
                            score += 15

                        # Solo aceptamos realmente el patrón DDDDDL
                        if len(frag) >= 5 and frag[4] == "9":
                            score += 85
                        elif len(frag) >= 5 and frag[4] in ["P", "R"]:
                            score += 35
                        else:
                            score -= 250

                    # Si el quinto carácter es claramente 9, penalizar DDDDLL porque

                    if mascara == "DDDDLL" and frag[4] == "9":
                        # Caso contextual: 3138-BP suele leerse como 3138-9P.
                        if len(frag) >= 6 and frag[5] in ["P", "R", "B"]:
                            score -= 5
                            if cand[4] == "B":
                                score += 28
                        else:
                            score -= 70
                    elif mascara == "DDDDLL" and frag[4] in ["P", "R"]:
                        score -= 70
                    elif mascara == "DDDDLL" and frag[4].isdigit():
                        score -= 10

                    # Evitar que dos dígitos finales claros se conviertan en letras.
                    if mascara == "DDDDLL" and frag[4].isdigit() and frag[5].isdigit():
                        score -= 80

                    # Para formatos que empiezan con letras, penalizar si arrancan con números claros.
                    if mascara.startswith("LL"):
                        primeros_letra = frag[:mascara.count("L")]
                        digitos_claros = sum(1 for c in primeros_letra if c.isdigit())
                        score -= digitos_claros * 15

                    #refuerzo contextual para placas antiguas mixtas tipo P7-2478.
                    if mascara == "LDDDDD":
                        if cand[0].isalpha() and cand[1:].isdigit():
                            score += 35
                            # P suele ser frecuente en estas placas antiguas/motos.
                            if cand[0] in ["P", "B", "R"]:
                                score += 12
                        if frag[0].isdigit():
                            score -= 25

                    if is_celeste:
                        if tipo in ["LDDDDD", "LLDDDDD", "LLLDDD", "AUTO LIVIANO", "MOTO ANTIGUA MIXTA"]:
                            score -= 200
                        
                        # Regla: Priorizar lectura directa si ya coincide con el formato
                        coincide_exacto = False
                        if len(frag) == len(mascara):
                            coincide_exacto = all(
                                (m == 'D' and c.isdigit()) or (m == 'L' and c.isalpha())
                                for m, c in zip(mascara, frag)
                            )
                        
                        if coincide_exacto and mascara in ["DDDDLL", "DDDDDL", "LDDDDD"]:
                            if mutaciones == 0:
                                score += 200
                                if info["tiene_strip"]:
                                    score += 200 # Bono extra masivo por coincidencia perfecta viniendo del strip
                            else:
                                score -= 150
                                
                        if mascara == "DDDDLL" and info["tiene_strip"] and "letras_negras" in "".join(info["fuentes"]):
                            score += 100
                        if mascara == "DDDDLL" and cand[-1] == "P":
                            score += 40 # P es sumamente frecuente al final de motos DDDDLL, desempatar a favor de P
                            
                        # Regla suave para ruido lateral en el strip que convierte P en U
                        if mascara == "DDDDLL" and cand[-1] == "P" and len(frag) >= 6 and frag[-2:] in ["LU", "HU", "YU"]:
                            fuentes_texto = "".join(info["fuentes"])
                            fuente_confiable = any(f in fuentes_texto for f in [
                                "strip_letras_negras", "strip_sin_bordes_laterales", "zona_celeste", "zona_clahe_celeste"
                            ])
                            # Solo si los primeros 4 caracteres son dígitos claros en la lectura original
                            primeros_4_digitos = sum(1 for c in frag[:4] if c.isdigit())
                            if fuente_confiable and primeros_4_digitos >= 3:
                                score += 80  # Bonus compensatorio para vencer la penalización por mutación U -> P
                                
                        if mascara == "DDDDDL" and len(frag) >= 5 and frag[4] == "9" and info["tiene_strip"]:
                            score += 80

                    evaluados.append({
                        "score": score,
                        "placa": formatear(cand),
                        "tipo": tipo,
                        "candidato": cand,
                        "fragmento": frag,
                        "lectura": lectura_limpia,
                        "mutaciones": mutaciones,
                        "fuentes": ",".join(sorted(info["fuentes"])),
                        "count": info["count"],
                        "tiene_strip": info["tiene_strip"],
                    })

    # Tie-break: score, si vino del strip, menos mutaciones, más repeticiones.
    evaluados.sort(
        key=lambda x: (
            x["score"],
            1 if x["tiene_strip"] else 0,
            -x["mutaciones"],
            x["count"],
        ),
        reverse=True,
    )

    mejor_alfanum = evaluados[0] if evaluados else None
    mejor_num = numericos_fallback[0] if numericos_fallback else None

    candidatos_finales = []
    if mejor_alfanum is not None:
        candidatos_finales.append(mejor_alfanum)
    if mejor_num is not None:
        candidatos_finales.append(mejor_num)
        
    candidatos_finales.sort(key=lambda x: x["score"], reverse=True)
    
    mejor = candidatos_finales[0] if candidatos_finales else None
    lista_debug = evaluados + numericos_fallback

    if mejor is None:
        return False, "", "No se obtuvo texto OCR útil", -1, []

    print("\n[DEBUG VALIDACION] TOP 12 CANDIDATOS")
    for e in lista_debug[:12]:
        print(
            f"{e['placa']:10s} | {e['tipo']:18s} | "
            f"score={e['score']:5.1f} | mut={e['mutaciones']} | rep={e['count']} | "
            f"strip={e['tiene_strip']} | frag={e['fragmento']} | fuentes={e['fuentes']}"
        )

    if is_celeste:
        try:
            import pandas as pd
            import os
            os.makedirs("results/debug", exist_ok=True)
            df_cand = pd.DataFrame(lista_debug)
            df_cand.to_csv("results/debug/candidatos_celeste.csv", index=False)
        except Exception as e:
            print(f"[WARN] No se pudo guardar candidatos_celeste.csv: {e}")

    # Umbrales separados: alfanuméricos aceptan 110; numéricos requieren más evidencia.
    umbral = 155 if mejor["tipo"] == "MOTO NUMERICA" else 110
    if mejor["score"] < umbral:
        return False, mejor["placa"], f"Inválido / score bajo: {mejor['score']:.1f}", mejor["score"], lista_debug

    return True, mejor["placa"], mejor["tipo"], mejor["score"], lista_debug

def validacion_extendida(texto_crudo, is_celeste=False):
    return validar_pool_lecturas([texto_crudo], is_celeste=is_celeste)

