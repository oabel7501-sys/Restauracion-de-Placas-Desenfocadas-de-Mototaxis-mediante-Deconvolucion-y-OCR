
try:
    import pytesseract
except ImportError:
    pytesseract = None
try:
    import rawpy
except ImportError:
    rawpy = None
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, simpledialog
import os
import glob
try:
    import easyocr
except ImportError:
    easyocr = None
from utils.config import DEBUG_GUARDAR_IMAGENES
from utils.image_io import cargar_imagen
from utils.roi import extraer_roi_plano
from utils.pipeline import procesar_fragmento
from utils.auto_crop import detectar_placa_celeste
from utils.excel_report import guardar_reporte_excel

class AppPlacas:
    def __init__(self, root):
        self.root = root
        self.root.title("Restauración y OCR de Placas Desenfocadas")
        self.root.configure(bg="#2c3e50")

        print("Cargando modelo OCR... por favor espera.")
        self.reader = None
        if easyocr is not None:
            try:
                self.reader = easyocr.Reader(["es", "en"], gpu=False, verbose=False)
                print("EasyOCR cargado correctamente.")
            except Exception as e:
                print(f"[WARN] EasyOCR no se pudo cargar: {e}")
        else:
            print("[WARN] EasyOCR no está instalado. Se usará Tesseract si está disponible.")

        if pytesseract is None and self.reader is None:
            messagebox.showwarning(
                "OCR no disponible",
                "No se encontró EasyOCR ni pytesseract. Instala al menos uno para leer placas."
            )

        lbl_titulo = tk.Label(
            self.root,
            text="Restauración y OCR de Placas Desenfocadas",
            font=("Helvetica", 20, "bold"),
            bg="#2c3e50",
            fg="#ecf0f1",
        )
        lbl_titulo.pack(pady=10)

        btn_subir = tk.Button(
            self.root,
            text="Subir foto y seleccionar placa",
            font=("Helvetica", 12),
            bg="#3498db",
            fg="white",
            cursor="hand2",
            command=self.cargar_imagen,
        )
        btn_subir.pack(pady=5)

        self.btn_auto_crop = tk.Button(
            self.root,
            text="Detectar placa celeste automáticamente",
            font=("Helvetica", 12),
            bg="#8e44ad",
            fg="white",
            cursor="hand2",
            command=self.auto_detectar_placa,
            state=tk.DISABLED
        )
        self.btn_auto_crop.pack(pady=5)

        self.btn_guardar_caso = tk.Button(
            self.root,
            text="Guardar caso de prueba",
            font=("Helvetica", 12),
            bg="#27ae60",
            fg="white",
            cursor="hand2",
            command=self.guardar_caso_de_prueba,
            state=tk.DISABLED
        )
        self.btn_guardar_caso.pack(pady=5)
        
        self.frame_excel = tk.Frame(self.root, bg="#2c3e50")
        self.frame_excel.pack(pady=5)
        
        lbl_placa = tk.Label(self.frame_excel, text="Placa corregida / placa real:", bg="#2c3e50", fg="white", font=("Helvetica", 11))
        lbl_placa.grid(row=0, column=0, padx=5)
        
        self.txt_placa_corregida = tk.Entry(self.frame_excel, font=("Helvetica", 12), width=15)
        self.txt_placa_corregida.grid(row=0, column=1, padx=5)
        
        self.btn_guardar_excel = tk.Button(
            self.frame_excel,
            text="Guardar resultado en Excel",
            font=("Helvetica", 10, "bold"),
            bg="#f39c12",
            fg="white",
            cursor="hand2",
            command=self.guardar_excel,
            state=tk.DISABLED
        )
        self.btn_guardar_excel.grid(row=0, column=2, padx=10)

        self.lbl_resultado = tk.Label(
            self.root,
            text="Sube una foto para analizar...",
            font=("Helvetica", 14),
            bg="#2c3e50",
            fg="#f1c40f",
        )
        self.lbl_resultado.pack(pady=5)

        self.canvas_frame = tk.Frame(self.root, bg="#34495e")
        self.canvas_frame.pack(pady=5)
        self.canvas = tk.Canvas(self.canvas_frame, bg="#34495e", cursor="crosshair", highlightthickness=0)
        self.canvas.pack()

        self.frame_etapas = tk.Frame(self.root, bg="#2c3e50")
        self.lbls_etapas = {}

        nombres = [
            ("roi", "1. ROI Aplanado"),
            ("preprocesada", "2. Preprocesada (CLAHE)"),
            ("binarizada", "3. Binarizada"),
            ("segmentacion", "4. Segmentación"),
            ("caracteres", "5. Caracteres Extraídos"),
            ("placa", "6. Placa Detectada"),
        ]

        for i, (key, title) in enumerate(nombres):
            row = i // 3
            col = i % 3
            frame_celda = tk.Frame(self.frame_etapas, bg="#34495e", bd=2, relief=tk.GROOVE)
            frame_celda.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

            lbl_tit = tk.Label(
                frame_celda,
                text=title,
                font=("Helvetica", 10, "bold"),
                bg="#34495e",
                fg="#ecf0f1",
            )
            lbl_tit.pack(side=tk.TOP, pady=2)

            lbl_img = tk.Label(frame_celda, bg="#2c3e50", width=20, height=6)
            lbl_img.pack(side=tk.BOTTOM, padx=5, pady=5, expand=True)
            self.lbls_etapas[key] = lbl_img

        self.puntos_roi = []
        self.original_img_bgr = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.ruta_actual = ""
        self.img_tk = None

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)

    def cargar_imagen(self):
        ruta_imagen = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=[("Archivos de Imagen", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp *.arw *.ARW")],
        )
        if not ruta_imagen:
            return

        self.frame_etapas.pack_forget()
        self.btn_guardar_caso.config(state=tk.DISABLED)
        self.btn_auto_crop.config(state=tk.DISABLED)
        self.ruta_actual = ruta_imagen
        self.lbl_resultado.configure(
            text="Haz clic en 'Detectar placa' o selecciona las 4 esquinas manualmente.",
            fg="#f39c12",
        )
        self.root.update()

        try:
            if ruta_imagen.lower().endswith(".arw"):
                if rawpy is None:
                    raise RuntimeError("rawpy no está instalado. Instala con: pip install rawpy")
                with rawpy.imread(ruta_imagen) as raw:
                    rgb = raw.postprocess()
                self.original_img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                img_pil = Image.fromarray(rgb)
            else:
                self.original_img_bgr = cv2.imread(ruta_imagen)
                if self.original_img_bgr is None:
                    raise RuntimeError("OpenCV no pudo leer la imagen. Revisa la ruta o el formato.")
                img_rgb = cv2.cvtColor(self.original_img_bgr, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)

            canvas_max_width = 700
            canvas_max_height = 350
    
            orig_w, orig_h = img_pil.size

            ratio = min(canvas_max_width / orig_w, canvas_max_height / orig_h)
            new_w = max(1, int(orig_w * ratio))
            new_h = max(1, int(orig_h * ratio))

            self.scale_x = orig_w / new_w
            self.scale_y = orig_h / new_h

            img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.img_tk = ImageTk.PhotoImage(img_resized)

            self.canvas.config(width=new_w, height=new_h)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
            self.puntos_roi = []
            
            self.btn_auto_crop.config(state=tk.NORMAL)
            self.btn_guardar_excel.config(state=tk.DISABLED)
            self.metodo_recorte_actual = "MANUAL"

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{e}")

    def on_button_press(self, event):
        if self.original_img_bgr is None:
            return

        x_orig = int(event.x * self.scale_x)
        y_orig = int(event.y * self.scale_y)
        self.puntos_roi.append((x_orig, y_orig))

        r = 4
        self.canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill="red", outline="red")

        if len(self.puntos_roi) > 1:
            prev_x = int(self.puntos_roi[-2][0] / self.scale_x)
            prev_y = int(self.puntos_roi[-2][1] / self.scale_y)
            self.canvas.create_line(prev_x, prev_y, event.x, event.y, fill="red", width=2)

        if len(self.puntos_roi) == 4:
            first_x = int(self.puntos_roi[0][0] / self.scale_x)
            first_y = int(self.puntos_roi[0][1] / self.scale_y)
            self.canvas.create_line(event.x, event.y, first_x, first_y, fill="red", width=2)
            self.procesar_perspectiva()
            self.puntos_roi = []

    def auto_detectar_placa(self):
        if self.original_img_bgr is None:
            return

        self.lbl_resultado.configure(text="Buscando placa celeste...", fg="#f39c12")
        self.root.update()

        recorte_plano, puntos, msg = detectar_placa_celeste(self.original_img_bgr)
        if recorte_plano is None:
            messagebox.showinfo("Atención", msg)
            self.lbl_resultado.configure(
                text=msg,
                fg="#f39c12"
            )
            return
            
        if "dudoso" in msg.lower():
            respuesta = messagebox.askyesno(
                "Recorte dudoso", 
                "El recorte automático es dudoso.\n¿Deseas usarlo de todos modos?\n\n(Sí = Usar automático, No = Hacer recorte manual)"
            )
            if not respuesta:
                self.lbl_resultado.configure(
                    text="Recorte automático descartado. Selecciona las 4 esquinas manualmente.",
                    fg="#f39c12"
                )
                return

        self.metodo_recorte_actual = "AUTOMATICO_CELESTE"
        
        # Dibujar rectangulo detectado (opcional)
        if puntos:
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
            for i in range(4):
                p1 = puntos[i]
                p2 = puntos[(i+1)%4]
                self.canvas.create_line(p1[0]*self.scale_x, p1[1]*self.scale_y, 
                                        p2[0]*self.scale_x, p2[1]*self.scale_y, 
                                        fill="green", width=3)

        self.lbl_resultado.configure(text="Procesando restauración + OCR robusto...", fg="#f39c12")
        self.root.update()

        self.recorte_plano_actual = recorte_plano
        self.tiempo_inicio = cv2.getTickCount()
        etapas = procesar_fragmento(recorte_plano, self.reader, is_celeste=True)
        tiempo_fin = cv2.getTickCount()
        self.tiempo_proceso = (tiempo_fin - self.tiempo_inicio) / cv2.getTickFrequency()

        self.ultima_placa_detectada = etapas.get("placa", "")
        self.ultimo_tipo_detectado = etapas.get("tipo", "MOTO_INVERTIDO")
        self.ultimo_score = etapas.get("score", 0)
        self.ultimo_valido = etapas.get("valido", False)
        self.btn_guardar_caso.config(state=tk.NORMAL)

        self.frame_etapas.pack(pady=10)
        self.actualizar_panel_etapas(etapas)

    def procesar_perspectiva(self):
        if len(self.puntos_roi) != 4:
            return

        self.metodo_recorte_actual = "MANUAL"
        recorte_plano = extraer_roi_plano(self.original_img_bgr, self.puntos_roi)
        if recorte_plano is None:
            messagebox.showwarning("Error ROI", "No se pudo extraer la ROI con los puntos dados.")
            return

        if DEBUG_GUARDAR_IMAGENES:
            cv2.imwrite("roi_aplanado_debug.jpg", recorte_plano)

        self.lbl_resultado.configure(text="Procesando restauración + OCR robusto...", fg="#f39c12")
        self.root.update()

        self.recorte_plano_actual = recorte_plano
        self.tiempo_inicio = cv2.getTickCount()
        # En recorte manual, también aplicamos is_celeste=True como opción adicional
        etapas = procesar_fragmento(recorte_plano, self.reader, is_celeste=True)
        tiempo_fin = cv2.getTickCount()
        self.tiempo_proceso = (tiempo_fin - self.tiempo_inicio) / cv2.getTickFrequency()

        self.ultima_placa_detectada = etapas.get("placa", "")
        self.ultimo_tipo_detectado = etapas.get("tipo", "MOTO_INVERTIDO")
        self.ultimo_score = etapas.get("score", 0)
        self.ultimo_valido = etapas.get("valido", False)
        self.btn_guardar_caso.config(state=tk.NORMAL)

        self.frame_etapas.pack(pady=10)
        self.actualizar_panel_etapas(etapas)

    def actualizar_panel_etapas(self, etapas):
        def preparar_img_gui(img, size=(100, 100)):
            if img is None:
                return None
            if isinstance(img, list):
                return None
            if getattr(img, "size", 0) == 0:
                return None

            if len(img.shape) == 2:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_pil = Image.fromarray(img_rgb)
            img_pil.thumbnail(size)
            return ImageTk.PhotoImage(img_pil)

        for key in ["roi", "preprocesada", "binarizada", "segmentacion"]:
            if key in etapas and etapas[key] is not None:
                tk_img = preparar_img_gui(etapas[key])
                if tk_img is not None:
                    self.lbls_etapas[key].configure(image=tk_img, text="", width=0, height=0)
                    self.lbls_etapas[key].image = tk_img

        if "caracteres" in etapas and len(etapas["caracteres"]) > 0:
            try:
                chars = [
                    cv2.copyMakeBorder(c, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
                    for c in etapas["caracteres"]
                ]
                alto_max = max(c.shape[0] for c in chars)
                chars_resized = []
                for c in chars:
                    nuevo_w = max(1, int(c.shape[1] * alto_max / c.shape[0]))
                    chars_resized.append(cv2.resize(c, (nuevo_w, alto_max), interpolation=cv2.INTER_CUBIC))
                strip = np.hstack(chars_resized)
                tk_strip = preparar_img_gui(strip, size=(100, 90))
                self.lbls_etapas["caracteres"].configure(image=tk_strip, text="", width=0, height=0)
                self.lbls_etapas["caracteres"].image = tk_strip
            except Exception as e:
                self.lbls_etapas["caracteres"].configure(
                    image="",
                    text=f"Error panel chars:\n{e}",
                    width=24,
                    height=6,
                    fg="#ecf0f1",
                )
        else:
            self.lbls_etapas["caracteres"].configure(
                image="",
                text="N/A\nSin caracteres claros",
                width=20,
                height=6,
                fg="#ecf0f1",
            )

        placa = etapas.get("placa", "")
        es_valido = etapas.get("valido", False)
        tipo = etapas.get("tipo", "")
        score = etapas.get("score", 0)

        if es_valido:
            self.lbls_etapas["placa"].configure(
                text=f"{placa}\n\n({tipo})\nScore: {score:.0f}",
                font=("Helvetica", 12, "bold"),
                fg="#2ecc71",
                image="",
                width=18,
                height=5,
            )
            self.lbl_resultado.configure(text=f"PLACA RECUPERADA: {placa} | TIPO: {tipo} | SCORE: {score:.0f}", fg="#2ecc71")
            # Guardar solo si hay confianza suficiente. Puedes bajar este umbral si deseas.
            # (El guardado en base de datos ha sido eliminado a favor de Excel)
        else:
            texto_mostrar = "No Detectada"
            if placa:
                texto_mostrar += f"\n\nMejor aprox.:\n{placa}"
            self.lbls_etapas["placa"].configure(
                text=texto_mostrar,
                font=("Helvetica", 10),
                fg="#e74c3c",
                image="",
                width=18,
                height=5,
            )
            self.lbl_resultado.configure(text=f"RECHAZADA o DUDOSA: {placa}", fg="#e74c3c")
            
        # Preparar data para el Excel
        self.ultimo_resultado_excel = {
            "placa_detectada": placa,
            "placa_corregida": placa,
            "tipo_detectado": tipo,
            "valido": es_valido,
            "metodo_color": "HSV celeste" if getattr(self, "metodo_recorte_actual", "") == "AUTOMATICO_CELESTE" else "N/A",
            "ruta_strip": "results/debug/strip_caracteres.png",
            "observacion": "" if es_valido else "No se obtuvo texto OCR útil",
        }
        
        self.txt_placa_corregida.delete(0, tk.END)
        self.txt_placa_corregida.insert(0, placa)
        self.btn_guardar_excel.config(state=tk.NORMAL)
        
    def guardar_excel(self):
        if not hasattr(self, "ultimo_resultado_excel"):
            return
            
        placa_corregida = self.txt_placa_corregida.get().strip()
        self.ultimo_resultado_excel["placa_corregida"] = placa_corregida
        
        try:
            ruta = guardar_reporte_excel(self.ultimo_resultado_excel)
            messagebox.showinfo("Excel", f"Guardado exitosamente en:\n{ruta}")
            self.btn_guardar_excel.config(state=tk.DISABLED)
        except Exception as e:
            messagebox.showerror("Error", f"Fallo al guardar Excel:\n{e}")

    def guardar_caso_de_prueba(self):
        if not hasattr(self, 'recorte_plano_actual') or self.recorte_plano_actual is None:
            return
            
        os.makedirs("data/samples", exist_ok=True)
        existentes = glob.glob("data/samples/placa_*.jpg")
        numeros = []
        for f in existentes:
            base = os.path.basename(f)
            try:
                num = int(base.replace("placa_", "").replace(".jpg", ""))
                numeros.append(num)
            except ValueError:
                pass
        
        siguiente = max(numeros) + 1 if numeros else 1
        nuevo_nombre = f"placa_{siguiente:03d}.jpg"
        ruta_guardado = f"data/samples/{nuevo_nombre}"
        
        placa_gt = simpledialog.askstring(
            "Confirmar Placa Real", 
            "Edita o confirma la placa real (Ground Truth):", 
            initialvalue=getattr(self, 'ultima_placa_detectada', '')
        )
        
        if not placa_gt:
            return
            
        tipo_gt = getattr(self, 'ultimo_tipo_detectado', 'MOTO_INVERTIDO')
        if not tipo_gt:
            tipo_gt = "MOTO_INVERTIDO"
            
        cv2.imwrite(ruta_guardado, self.recorte_plano_actual)
        
        csv_path = "tests/test_dataset.csv"
        if not os.path.exists(csv_path):
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("image_path,plate_gt,tipo,observacion\n")
                
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(f"data/samples/{nuevo_nombre},{placa_gt.upper()},{tipo_gt},guardado desde GUI\n")
            
        # Guardar en reporte Excel
        import datetime
        datos_excel = {
            "fecha_hora": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "imagen_original": os.path.basename(self.ruta_actual),
            "metodo_recorte": getattr(self, "metodo_recorte_actual", "MANUAL"),
            "placa_detectada": getattr(self, "ultima_placa_detectada", ""),
            "tipo_detectado": getattr(self, "ultimo_tipo_detectado", ""),
            "score": getattr(self, "ultimo_score", 0),
            "valido": getattr(self, "ultimo_valido", False),
            "tiempo_s": getattr(self, "tiempo_proceso", 0),
            "metodo_color": "HSV_V" if getattr(self, "metodo_recorte_actual", "") == "AUTOMATICO_CELESTE" else "GRIS_ESTANDAR",
            "metodo_morfologico": "CIERRE_APERTURA",
            "ruta_roi": ruta_guardado,
            "ruta_mascara_celeste": "results/debug/mascara_celeste.png",
            "ruta_binarizada": "results/debug/binarizada_letras_negras.png",
            "observacion": f"Caso guardado gui: {placa_gt}"
        }
        guardar_reporte_excel(datos_excel)
            
        messagebox.showinfo("Éxito", f"Caso de prueba guardado en {ruta_guardado}\ny agregado al dataset y reporte excel.")

if __name__ == '__main__':
    root = tk.Tk()
    app = AppPlacas(root)
    root.mainloop()
