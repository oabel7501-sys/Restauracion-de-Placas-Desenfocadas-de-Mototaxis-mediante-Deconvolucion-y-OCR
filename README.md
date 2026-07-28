# Restauración de Placas Desenfocadas de Mototaxis mediante Deconvolución y OCR

##  Descripción General
Este proyecto es una aplicación de escritorio desarrollada en Python enfocada en la visión por computadora. Su propósito principal es procesar, restaurar y reconocer los caracteres (OCR) de placas vehiculares —con un enfoque especial en mototaxis— que han sido capturadas con desenfoque, distorsión de perspectiva o baja calidad. El sistema cuenta con una interfaz gráfica (GUI) que permite realizar el recorte manual o la detección automática mediante espacios de color, generando reportes automatizados en Excel.

##  El Problema
En el ámbito de la seguridad ciudadana y el control vial, es común que las cámaras de seguridad o los dispositivos móviles de transeúntes capten imágenes de vehículos involucrados en incidentes. Sin embargo, debido al movimiento rápido, ángulos pronunciados o baja resolución, las placas (especialmente las celestes de mototaxis) resultan borrosas o ilegibles. Esto dificulta enormemente la identificación vehicular y retrasa los procesos de investigación.

##  Solución Propuesta
Se ha desarrollado un software estructurado de manera modular que integra algoritmos de procesamiento de imágenes y reconocimiento óptico de caracteres. La aplicación permite al usuario:
1. Cargar una imagen y detectar automáticamente la placa (basado en el color celeste) o seleccionarla manualmente mediante 4 puntos para corregir la perspectiva.
2. Aplicar un pipeline de preprocesamiento que incluye ecualización de histogramas (CLAHE), binarización morfológica y segmentación de caracteres.
3. Extraer el texto limpio utilizando motores de OCR (EasyOCR / Tesseract).
4. Guardar los casos de prueba y exportar los resultados directamente a una hoja de cálculo para llevar un registro eficiente.

##  Objetivos
*   **Principal:** Desarrollar un sistema de software funcional capaz de restaurar visualmente y extraer texto de placas vehiculares desenfocadas o distorsionadas.
*   **Específicos:**
    *   Implementar una interfaz gráfica amigable e intuitiva para usuarios no técnicos.
    *   Modularizar el código fuente para garantizar su escalabilidad y facilitar su reutilización en futuros proyectos académicos o de ingeniería.
    *   Automatizar la generación de reportes estructurados.

##  Limitaciones
*   **Desenfocado Extremo:** El sistema no puede recuperar caracteres cuando la pérdida de información de los píxeles en la imagen original es absoluta (irreversible).
*   **Inclinación Excesiva:** Si el ángulo de la placa es demasiado extremo, la corrección de perspectiva matemática (aplanamiento) deforma la morfología de las letras, provocando fallos en la lectura del OCR.

##  Estructura del Repositorio
El código está organizado modularmente para separar la interfaz gráfica de la lógica de procesamiento y visión por computadora:

*   `app.py`: Archivo principal que contiene la interfaz gráfica de usuario y la orquestación del programa.
*   `utils/`: Directorio que contiene todas las librerías y módulos específicos del procesamiento.
    *   `__init__.py`: Inicializador del módulo.
    *   `auto_crop.py`: Lógica para la detección automática de placas mediante algoritmos de color.
    *   `config.py`: Variables de configuración y constantes globales.
    *   `debug_tools.py`: Herramientas para visualizar y depurar etapas intermedias de las imágenes.
    *   `excel_report.py`: Módulo encargado de la escritura y exportación de datos.
    *   `image_io.py`: Funciones para lectura, escritura y manejo de formatos (incluyendo RAW).
    *   `image_utils.py`: Transformaciones y ajustes de imagen de uso general.
    *   `ocr_utils.py`: Integración y manejo de EasyOCR y Pytesseract.
    *   `pipeline.py`: Define el flujo secuencial de restauración y procesamiento.
    *   `restoration.py`: Algoritmos específicos (como CLAHE y filtros) para mejorar la nitidez.
    *   `roi.py`: Extracción y corrección de perspectiva de la Región de Interés.
    *   `segmentation.py`: Búsqueda de contornos y aislamiento individual de los caracteres.
    *   `validation.py`: Reglas lógicas y expresiones regulares para validar formatos de placas.

##  Requisitos y Entorno de Trabajo
Para garantizar que el procesamiento de imágenes (especialmente el OCR y las transformaciones matriciales) se ejecute de forma fluida y sin cuellos de botella, se recomienda el siguiente entorno:

*   **Sistema Operativo:** Windows 11 (Probado y validado en este entorno).
*   **Hardware Mínimo:** 8 GB de Memoria RAM.
*   **Lenguaje:** Python 3.9+
*   **Dependencias y Paquetes Principales:**
    *   `opencv-python` (cv2)
    *   `easyocr`
    *   `pytesseract`
    *   `numpy`
    *   `Pillow` (PIL)
    *   `rawpy` (Para lectura de imágenes en formato crudo/ARW)

*(Nota: Para instalar las dependencias, se recomienda ejecutar `pip install -r requirements.txt`)*

##  Dataset
El sistema incluye la capacidad de generar su propio conjunto de datos localmente. A través de la interfaz, el operador puede procesar imágenes reales, confirmar el "Ground Truth" (texto real de la placa) y guardar el caso. Estos recortes y sus metadatos se almacenan en `data/samples/` y `tests/test_dataset.csv` respectivamente, formando un dataset propietario para futuras mejoras o entrenamientos. *(Nota: Si se planea hacer público este dataset de mototaxis posteriormente, el enlace de descarga se incluirá en esta sección).*

---
*Desarrollado como proyecto aplicativo por estudiantes de Ingeniería Electrónica y Telecomunicaciones de la Universidad Nacional de Piura.*

##  Equipo de Desarrollo

*   BRUNO CORDOVA ANTONELLA STEFANY
*   MIJAHUANGA JIMÉNEZ JHEYLER
*   PEDEMONTE TIMANA CRISTHIAN JOSUE
*   PINGO UMBO JERSSON YAIR
*   SAAVEDRA CARRILLO GIANCARLO GUSTAVO
*   VILELA OROZCO KEVIN ABEL
