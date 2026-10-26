import re, json
import cv2, fitz, nltk, easyocr
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords

st.set_page_config(page_title="Análisis de Recibos Peruanos", page_icon="📄", layout="wide")

# ---------- Modelos (se cargan una sola vez) ----------
@st.cache_resource
def cargar_modelos():
    nltk.download('stopwords', quiet=True)
    return easyocr.Reader(['es'], gpu=False), set(stopwords.words('spanish'))

reader, STOP_ES = cargar_modelos()

# ---------- Carga y preprocesamiento ----------
def cargar_imagen(archivo):
    datos = archivo.read()
    if archivo.name.lower().endswith('.pdf'):
        pix = fitz.open(stream=datos, filetype='pdf')[0].get_pixmap(dpi=200)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR)
    return cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)

def preprocesar(img_bgr):
    h, w = img_bgr.shape[:2]
    escala = 1600 / max(h, w)                                   # redimensionamiento
    img = cv2.resize(img_bgr, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                # escala de grises
    gris = cv2.fastNlMeansDenoising(gris, None, 10, 7, 21)      # eliminar ruido
    gris = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gris)   # contraste
    return gris

# ---------- OCR y limpieza ----------
def hacer_ocr(img_proc):
    res = reader.readtext(img_proc, detail=1)
    texto = ' '.join(r[1] for r in res)
    conf = float(np.mean([r[2] for r in res])) if res else 0.0
    return texto, conf

def limpiar_texto(texto):
    t = re.sub(r'[^a-záéíóúüñA-ZÁÉÍÓÚÜÑ0-9\s.,\-]', ' ', texto)
    t = re.sub(r'\s+', ' ', t).strip().lower()
    tokens = [w for w in re.findall(r'[a-záéíóúüñ0-9]+', t) if w not in STOP_ES and len(w) > 2]
    return t, ' '.join(tokens)

# ---------- NLP (igual que el Paso 7 del notebook) ----------
EMPRESAS = {
    'luz':      ['luz del sur','pluz', 'enel', 'hidrandina', 'electronoroeste', 'electro oriente',
                 'electrocentro', 'seal', 'electro sur'],
    'agua':     ['sedapal', 'sedalib', 'epsel', 'sedacusco', 'seda ayacucho', 'emapa'],
    'internet': ['movistar', 'claro', 'entel', 'bitel', 'wow', 'win', 'telefonica', 'telefónica'],
}
PALABRAS_TIPO = {
    'luz':      ['kwh', 'energía', 'energia', 'eléctric', 'electric', 'osinergmin', 'potencia', 'suministro'],
    'agua':     ['m3', 'm³', 'alcantarillado', 'agua potable', 'sunass', 'sedapal', 'cargo fijo'],
    'internet': ['mbps', 'fibra', 'internet', 'megas', 'plan', 'router', 'telecomunicaciones'],
}

def clasificar_servicio(texto):
    t = texto.lower()
    puntos = {k: 0 for k in PALABRAS_TIPO}
    for tipo, empresas in EMPRESAS.items():
        puntos[tipo] += 3 * sum(e in t for e in empresas)
    for tipo, palabras in PALABRAS_TIPO.items():
        puntos[tipo] += sum(p in t for p in palabras)
    mejor = max(puntos, key=puntos.get)
    return mejor if puntos[mejor] > 0 else 'otro'

def detectar_empresa(texto):
    t = texto.lower()
    for empresas in EMPRESAS.values():
        for e in empresas:
            if re.search(rf'\b{re.escape(e)}\b', t):
                return e.title()
    return 'No detectada'

def a_numero(s):
    s = s.strip()
    ultimo = max(s.rfind('.'), s.rfind(','))
    if ultimo == -1:
        return float(s)
    return float(f"{re.sub(r'[.,]', '', s[:ultimo])}.{s[ultimo+1:]}")

def extraer_ruc(texto):
    m = re.search(r'RUC\W{0,10}(\d{11})', texto, re.IGNORECASE) or re.search(r'\b((?:10|20)\d{9})\b', texto)
    return m.group(1) if m else 'No detectado'

def extraer_monto_total(texto):
    # se prueba por prioridad: "total a pagar" primero
    for clave in [r'total\s*a\s*pagar', r'importe\s*total', r'monto\s*total']:
        m = re.search(clave + r'.{0,60}?(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})', texto, re.IGNORECASE)
        if m:
            return a_numero(m.group(1))
    montos = re.findall(r'\d{1,3}(?:[.,]\d{3})*[.,]\d{2}\b', texto)
    return max(a_numero(x) for x in montos) if montos else None

def extraer_fechas(texto):
    numericas = re.findall(r'\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b', texto)
    con_mes   = re.findall(r'\b[\dOo]{1,2}[/\-.][A-Za-z0]{3}[/\-.]\d{4}\b', texto)
    escritas  = re.findall(r'\b\d{1,2}\s+de\s+[a-záéíóú]+\s+(?:de\s+)?\d{4}\b', texto, re.IGNORECASE)
    return numericas + con_mes + escritas

def extraer_consumo(texto):
    m = re.search(r'(?<![\d.,])([1-9]\d*(?:[.,]\d+)?)\s*(kwh|m3|m³|mbps)', texto, re.IGNORECASE)
    return f"{m.group(1)} {m.group(2).lower()}" if m else 'No detectado'

def analizar_recibo(texto):
    return {'tipo_servicio': clasificar_servicio(texto), 'empresa': detectar_empresa(texto),
            'ruc': extraer_ruc(texto), 'fechas': extraer_fechas(texto),
            'monto_total': extraer_monto_total(texto), 'consumo': extraer_consumo(texto)}

# ---------- Interfaz ----------
st.title("📄 Análisis de Recibos Peruanos")
st.write("Sube recibos de luz, agua o internet (imagen o PDF) y obtén sus datos clave.")

archivos = st.file_uploader("Sube tus recibos", type=['jpg', 'jpeg', 'png', 'webp', 'bmp', 'pdf'],
                            accept_multiple_files=True)
registros, detalle = [], []

for archivo in archivos or []:
    with st.expander(f"📄 {archivo.name}", expanded=True):
        with st.spinner("Procesando..."):
            img = cargar_imagen(archivo)
            proc = preprocesar(img)
            texto, conf = hacer_ocr(proc)
            normalizado, limpio = limpiar_texto(texto)
            a = analizar_recibo(texto)

        c1, c2 = st.columns(2)
        c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
        c2.image(proc, caption="Preprocesada", use_container_width=True)

        st.subheader("Texto extraído")
        st.text_area("OCR", texto, height=150, key=f"ocr_{archivo.name}")
        st.subheader("Texto limpio")
        st.text_area("Limpio", limpio, height=100, key=f"limpio_{archivo.name}")

        st.subheader("Resultado del análisis")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Servicio", a['tipo_servicio'].title())
        m2.metric("Empresa", a['empresa'])
        m3.metric("Monto total", f"S/ {a['monto_total']}" if a['monto_total'] else "No detectado")
        m4.metric("Confianza OCR", f"{conf:.0%}")
        st.write(f"**RUC:** {a['ruc']}  |  **Consumo:** {a['consumo']}  |  **Fechas:** {', '.join(a['fechas'][:3]) or 'No detectadas'}")

    registros.append({'archivo': archivo.name, 'tipo_servicio': a['tipo_servicio'], 'empresa': a['empresa'],
                      'ruc': a['ruc'], 'fechas': ' | '.join(a['fechas']), 'monto_total': a['monto_total'],
                      'consumo': a['consumo'], 'confianza_ocr': round(conf, 4)})
    detalle.append({'archivo': archivo.name, 'texto_ocr': texto, 'texto_limpio': limpio, 'analisis': a})

if registros:
    st.divider()
    st.subheader("📋 Resumen de todos los recibos")
    df = pd.DataFrame(registros)
    st.dataframe(df, use_container_width=True)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Descargar CSV", df.to_csv(index=False).encode('utf-8-sig'),
                       "resultados_recibos.csv", "text/csv")
    d2.download_button("⬇️ Descargar JSON", json.dumps(detalle, ensure_ascii=False, indent=2, default=str),
                       "resultados_recibos.json", "application/json")
