# -*- coding: utf-8 -*-
import streamlit as st
from typing import List, Dict, Tuple
from math import ceil
import ast
import matplotlib.pyplot as plt
import numpy as np
from string import Template
from pathlib import Path
import io
import base64
import pandas as pd

Punto = Tuple[float, float, str, int]  # (x, y, tipo, id)

def cargar_datos_excel(uploaded_file):
    """
    Carga y procesa el archivo Excel del plan de cosecha.
    Retorna un DataFrame con los datos limpios.
    """
    try:
        # Leer Excel sin encabezados
        df = pd.read_excel(uploaded_file, header=None)
        
        # La fila 1 (índice 1) contiene los encabezados
        df.columns = df.iloc[1]
        
        # Eliminar las primeras dos filas (vacía y encabezados)
        df = df.iloc[2:].reset_index(drop=True)
        
        # Convertir columnas numéricas
        numeric_columns = ['CECO', 'CUARTEL', 'TOTAL PLANTAS', 'N° HILERAS', 
                          'ESTIMACION KG/HÁ', 'KILOS COSECHADOS FLOREO X PLANTA',
                          'KILOS COSECHADOS AL BARRER X PLANTA']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo Excel: {str(e)}")
        return None


def extraer_separaciones(marco_plantacion_str):
    """
    Extrae las separaciones entre hileras y árboles del texto de marco de plantación.
    Ejemplo: "4 entre hilera x 2 sobre hilera" -> (4.0, 2.0)
    """
    try:
        parts = marco_plantacion_str.lower().split('x')
        sep_hileras = float(parts[0].split('entre')[0].strip())
        sep_arboles = float(parts[1].split('sobre')[0].strip())
        return sep_hileras, sep_arboles
    except:
        return 4.0, 2.0  # Valores por defecto


def generar_mapa_cuadrado_txt(
    arboles_por_hilera: List[int],
    capacidad_bin_kg: float,
    kg_por_arbol: float,
    sep_hileras_m: float = 4.0,
    sep_arboles_m: float = 2.0,
    extra_caras: int = 2,
    output_path: str = "mapa_por_capacidad_cuadrado.txt",
) -> Tuple[str, str]:
    """
    Construye el mapa y lo guarda como .txt (claves numéricas sin comillas).
    Devuelve la ruta de salida y el contenido txt.
    """
    assert len(arboles_por_hilera) >= 1, "Debe haber al menos 1 hilera."
    n_hileras = len(arboles_por_hilera)

    # Centros de hileras (x)
    centros_x = [i * sep_hileras_m for i in range(n_hileras)]

    # Bin en x: entre hilera 2 y 3 si existen; si no, entre las disponibles
    if n_hileras >= 3:
        x_bin = (centros_x[1] + centros_x[2]) / 2.0
    elif n_hileras == 2:
        x_bin = (centros_x[0] + centros_x[1]) / 2.0
    else:  # n_hileras == 1
        x_bin = centros_x[0] + sep_hileras_m / 2.0

    # Precomputamos caras por árbol (dos caras por árbol, misma y)
    faces_by_hilera: List[List[Tuple[Tuple[float, float], Tuple[float, float]]]] = []
    for i, n_arboles in enumerate(arboles_por_hilera):
        cx = centros_x[i]
        xL, xR = cx - 0.5, cx + 0.5
        hilera_faces = []
        for j in range(n_arboles):
            y = j * sep_arboles_m
            hilera_faces.append(((xL, y), (xR, y)))
        faces_by_hilera.append(hilera_faces)

    # Secuencia round-robin por nivel (árbol j en todas las hileras, luego j+1, etc.)
    secuencia_caras: List[Tuple[float, float]] = []
    max_n = max(arboles_por_hilera)
    for j in range(max_n):
        for i in range(n_hileras):
            if j < arboles_por_hilera[i]:
                (xL, y), (xR, _) = faces_by_hilera[i][j]
                secuencia_caras.append((xL, y))  # cara izquierda
                secuencia_caras.append((xR, y))  # cara derecha

    # Cálculo de "caras por clase" en base a capacidad y kg/árbol (y +2 caras para cuadrar)
    kg_por_cara = kg_por_arbol / 2.0
    caras_min = ceil(capacidad_bin_kg / kg_por_cara)          # mínimo de caras para llenar
    caras_por_clase = caras_min + extra_caras                 # "cuadrado" (ej: +2 => 32)

    # Construcción de clases con tamaño fijo en caras (salvo última si no alcanza)
    clases: Dict[int, List[Punto]] = {}
    cara_id_global = 1
    idx = 0
    clase_idx = 1

    while idx < len(secuencia_caras):
        items: List[Punto] = []
        ys = []

        take = min(caras_por_clase, len(secuencia_caras) - idx)
        for _ in range(take):
            x, y = secuencia_caras[idx]
            items.append((x, y, 'cara', cara_id_global))
            ys.append(y)
            cara_id_global += 1
            idx += 1

        # Bin centrado en Y dentro de la clase
        y_bin = (min(ys) + max(ys)) / 2.0 if ys else 0.0
        items.append((x_bin, y_bin, 'bin', 1))

        clases[clase_idx] = items
        clase_idx += 1

    if not clases:
        clases[1] = [(x_bin, 0.0, 'bin', 1)]

    # Guardar como .txt con claves numéricas (sin comillas)
    def dict_to_text(d: Dict[int, List[Punto]]) -> str:
        lines = ["{"]
        for k in sorted(d.keys()):
            lines.append(f"  {k}: [")
            for (x, y, tipo, idx_item) in d[k]:
                lines.append(f"    [{x}, {y}, \"{tipo}\", {idx_item}],")
            lines.append("  ],")
        lines.append("}")
        return "\n".join(lines)

    txt = dict_to_text(clases)

    return output_path, txt


def plot_hileras_arboles_y_pasillos(
    arboles_por_hilera,
    sep_hileras_m,
    sep_arboles_m,
    HHOunB, primerHunB,
    kgBarrer, kgBarrerdos,
    num_clases=4,
    margen_x=1.0,
    dpi=180,
    titulo="Hileras, árboles y pasillos horizontales (1 bin/posición)",
    subtitulo=None,
    caption=None,
    draw_aisles=True,
    bins_y=None,
    bins_xy=None,
    bin_marker='o',
    bin_size=40,
    is_floreo=False
):
    n_hileras = len(arboles_por_hilera)
    if n_hileras == 0:
        raise ValueError("Debe haber al menos 1 hilera.")

    # X centers
    centros_x = [i * sep_hileras_m for i in range(n_hileras)]
    etiquetas_x = [f"Hilera {i+1}" for i in range(n_hileras)]

    # Aisles (1 bin/posición) if requested
    y_pasillos = [primerHunB + k * HHOunB for k in range(num_clases)] if draw_aisles else []

    # Y range
    if y_pasillos:
        y_max = y_pasillos[-1] + HHOunB * 0.5
    else:
        y_max = max(arboles_por_hilera) * sep_arboles_m
    y_min = 0.0

    fig, ax = plt.subplots(figsize=(11, 7))

    # Hileras y árboles
    for i, cx in enumerate(centros_x):
        ax.vlines(cx, y_min, y_max, linewidth=1, linestyles='dashed', color='gray', alpha=0.6)
        n_arboles = arboles_por_hilera[i]
        ys = [j * sep_arboles_m for j in range(n_arboles) if j * sep_arboles_m <= y_max]
        xs = [cx] * len(ys)
        ax.scatter(xs, ys, s=14, color='green', alpha=0.7)

    # Dibujar pasillos
    x0 = min(centros_x) - margen_x
    x1 = max(centros_x) + margen_x
    Y = []
    for idx, y in enumerate(y_pasillos, start=1):
        ax.hlines(y, x0, x1, linewidth=2, color='brown', alpha=0.5)
        ax.annotate(f"Pasillo {idx}, {y:.1f} m", xy=(x1, y), xytext=(5, 2),
                    textcoords="offset points", ha="left", va="bottom")
        Y.append(y)

    # Dibujar BINes según lista
    if not is_floreo:
        bins_y = Y

    if bins_xy:
        xs, ys = zip(*bins_xy) if len(bins_xy) else ([], [])
        if xs:
            ax.scatter(xs, ys, s=bin_size, marker=bin_marker, color='red', zorder=5)
    elif bins_y:
        if n_hileras >= 3:
            x_bin = (centros_x[1] + centros_x[2]) / 2.0
        elif n_hileras == 2:
            x_bin = (centros_x[0] + centros_x[1]) / 2.0
        else:
            x_bin = centros_x[0] + sep_hileras_m / 2.0
        xs = [x_bin] * len(bins_y)
        ax.scatter(xs, bins_y, s=bin_size, marker=bin_marker, color='red', zorder=5)
        if bins_y:
            ax.annotate("bines", xy=(x_bin, bins_y[0]), xytext=(0, 6),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9)

    # Ejes
    ax.set_xticks(centros_x, etiquetas_x, rotation=0)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Posición X (m)")
    ax.set_ylabel("Posición Y (m)")

    # Títulos
    if subtitulo is None:
        subtitulo = f"kg por árbol: {kgBarrer} | kg por cara: {kgBarrerdos}"
    fig.suptitle(titulo, fontsize=14, y=0.98)
    ax.set_title(subtitulo, fontsize=10, pad=8)

    # Caption
    if caption is None:
        caption = "Bines según lista de posiciones."
    fig.text(0.995, 0.01, caption, ha="right", va="bottom", fontsize=9)

    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.93])

    return fig


def main():
    st.set_page_config(page_title="Optimización de Cosecha - Lihueimo", layout="wide")

    st.title("🌳 Automatización de Documentos - Optimización de Cosecha")
    st.markdown("### Sistema de planificación para ubicación óptima de bines")

    # Sidebar con parámetros
    st.sidebar.header("⚙️ Configuración de Parámetros")

    # Opción para cargar Excel
    st.sidebar.subheader("📂 Cargar Plan de Cosecha")
    uploaded_file = st.sidebar.file_uploader(
        "Subir archivo Excel del plan de cosecha",
        type=['xlsx', 'xls'],
        help="Sube el archivo Excel con la información de los cuarteles"
    )

    # Variables para almacenar los datos
    df_cosecha = None
    
    if uploaded_file is not None:
        df_cosecha = cargar_datos_excel(uploaded_file)
        
        if df_cosecha is not None:
            st.sidebar.success("✅ Archivo cargado exitosamente!")
            
            # Selector de CECO y Cuartel
            st.sidebar.subheader("🎯 Seleccionar Cuartel")
            
            # Crear lista de opciones únicas
            df_cosecha['CECO_CUARTEL'] = df_cosecha['CECO'].astype(str) + ' - Cuartel ' + df_cosecha['CUARTEL'].astype(str)
            opciones = df_cosecha['CECO_CUARTEL'].tolist()
            
            seleccion = st.sidebar.selectbox(
                "Seleccionar CECO y Cuartel:",
                opciones,
                help="Selecciona el campo y cuartel a optimizar"
            )
            
            # Obtener datos del cuartel seleccionado
            idx = opciones.index(seleccion)
            fila = df_cosecha.iloc[idx]
            
            # Mostrar información del cuartel seleccionado
            st.sidebar.info(f"""
            **Cuartel seleccionado:**
            - CECO: {fila['CECO']}
            - Cuartel: {fila['CUARTEL']}
            - Total plantas: {fila['TOTAL PLANTAS']}
            - N° Hileras: {fila['N° HILERAS']}
            """)

    # Información del campo
    st.sidebar.subheader("📋 Información del Campo")
    
    if uploaded_file is not None and df_cosecha is not None:
        # Prellenar con datos del Excel
        campo = st.sidebar.text_input("Nombre del Campo", f"CECO {fila['CECO']} Cuartel {fila['CUARTEL']}")
        empresa = st.sidebar.text_input("Empresa", "Garcés")
        fecha = st.sidebar.text_input("Fecha", pd.Timestamp.now().strftime("%d/%m/%Y"))
    else:
        campo = st.sidebar.text_input("Nombre del Campo", "Lihueimo CECO 10784 Cuartel 1")
        empresa = st.sidebar.text_input("Empresa", "Garcés")
        fecha = st.sidebar.text_input("Fecha", "13/11/2025")

    # Parámetros de estructura
    st.sidebar.subheader("🌲 Estructura del Huerto")
    
    if uploaded_file is not None and df_cosecha is not None:
        # Calcular árboles por hilera automáticamente
        total_plantas = int(fila['TOTAL PLANTAS'])
        n_hileras = int(fila['N° HILERAS'])
        arboles_base = total_plantas // n_hileras
        arboles_extra = total_plantas % n_hileras
        
        # Crear lista de árboles por hilera
        arboles_list = [arboles_base] * n_hileras
        for i in range(arboles_extra):
            arboles_list[i] += 1
        
        arboles_str_default = str(arboles_list[-4:]) #ultimos 4 elementos
        
        # Extraer separaciones del marco de plantación
        sep_hileras_default, sep_arboles_default = extraer_separaciones(fila['MARCO PLANTACION'])
        
        # Obtener kg por árbol del Excel 
        kg_floreo_default = float(fila['KILOS COSECHADOS FLOREO X PLANTA'])
        kg_barrer_default = float(fila['KILOS COSECHADOS AL BARRER X PLANTA'])
    else:
        arboles_str_default = "[60,60,60,60]"
        sep_hileras_default = 4.0
        sep_arboles_default = 2.0
        kg_floreo_default = 2.8
        kg_barrer_default = 11.2

    arboles_str = st.sidebar.text_input(
        "Árboles por hilera (lista)",
        arboles_str_default,
        help="Lista de árboles en cada hilera. Se calcula automáticamente si cargas un Excel"
    )

    sep_hileras_m = st.sidebar.number_input(
        "Separación entre hileras (m)", 
        value=sep_hileras_default, 
        min_value=1.0, 
        max_value=10.0, 
        step=0.5
    )
    
    sep_arboles_m = st.sidebar.number_input(
        "Separación entre árboles (m)", 
        value=sep_arboles_default, 
        min_value=0.5, 
        max_value=5.0, 
        step=0.5
    )

    # Parámetros de cosecha
    st.sidebar.subheader("🍎 Parámetros de Cosecha")
    capacidad_bin_kg = st.sidebar.number_input(
        "Capacidad del bin (kg)", 
        value=216.0, 
        min_value=50.0, 
        max_value=1000.0, 
        step=10.0
    )
    
    kg_por_arbol_floreo = st.sidebar.number_input(
        "Kg por árbol (floreo)", 
        value=kg_floreo_default, 
        min_value=0.1, 
        max_value=100.0, 
        step=0.1
    )
    
    kg_por_arbol_barrer = st.sidebar.number_input(
        "Kg por árbol (barrer)", 
        value=kg_barrer_default, 
        min_value=0.1, 
        max_value=100.0, 
        step=0.1
    )

    extra_caras = st.sidebar.number_input(
        "Caras extra para cuadrar (barrer)", 
        value=2, 
        min_value=0, 
        max_value=10, 
        step=1
    )
    
    extra_caras_floreo = st.sidebar.number_input(
        "Caras extra para cuadrar (floreo)", 
        value=2, 
        min_value=0, 
        max_value=10, 
        step=1
    )

    # Botón de generar
    if st.sidebar.button("🚀 Generar Optimización", type="primary"):
        try:
            # Parse lista de árboles
            arboles_por_hilera = ast.literal_eval(arboles_str)
            assert isinstance(arboles_por_hilera, list) and all(isinstance(x, int) and x >= 0 for x in arboles_por_hilera)

            with st.spinner("Generando mapas y optimizaciones..."):
                # Generar mapas
                _, txt_floreo = generar_mapa_cuadrado_txt(
                    arboles_por_hilera=arboles_por_hilera,
                    capacidad_bin_kg=capacidad_bin_kg,
                    kg_por_arbol=kg_por_arbol_floreo,
                    sep_hileras_m=sep_hileras_m,
                    sep_arboles_m=sep_arboles_m,
                    extra_caras=extra_caras_floreo,
                    output_path="floreo.txt",
                )

                _, txt_barrer = generar_mapa_cuadrado_txt(
                    arboles_por_hilera=arboles_por_hilera,
                    capacidad_bin_kg=capacidad_bin_kg,
                    kg_por_arbol=kg_por_arbol_barrer,
                    sep_hileras_m=sep_hileras_m,
                    sep_arboles_m=sep_arboles_m,
                    extra_caras=extra_caras,
                    output_path="barrer.txt",
                )

                _, txt_2_B = generar_mapa_cuadrado_txt(
                    arboles_por_hilera=arboles_por_hilera,
                    capacidad_bin_kg=capacidad_bin_kg*2,
                    kg_por_arbol=kg_por_arbol_barrer,
                    sep_hileras_m=sep_hileras_m,
                    sep_arboles_m=sep_arboles_m,
                    extra_caras=extra_caras*2,
                    output_path="barrer_2B.txt",
                )

                _, txt_3_B = generar_mapa_cuadrado_txt(
                    arboles_por_hilera=arboles_por_hilera,
                    capacidad_bin_kg=capacidad_bin_kg*3,
                    kg_por_arbol=kg_por_arbol_barrer,
                    sep_hileras_m=sep_hileras_m,
                    sep_arboles_m=sep_arboles_m,
                    extra_caras=extra_caras*3,
                    output_path="barrer_3B.txt",
                )

                _, txt_4_B = generar_mapa_cuadrado_txt(
                    arboles_por_hilera=arboles_por_hilera,
                    capacidad_bin_kg=capacidad_bin_kg*4,
                    kg_por_arbol=kg_por_arbol_barrer,
                    sep_hileras_m=sep_hileras_m,
                    sep_arboles_m=sep_arboles_m,
                    extra_caras=extra_caras*4,
                    output_path="barrer_4B.txt",
                )

                # Parse resultados
                d_barrer = ast.literal_eval(txt_barrer)
                d_floreo = ast.literal_eval(txt_floreo)
                d_barrer_2 = ast.literal_eval(txt_2_B)
                d_barrer_3 = ast.literal_eval(txt_3_B)
                d_barrer_4 = ast.literal_eval(txt_4_B)

                # Calcular métricas
                kgBarrerdos = kg_por_arbol_barrer / 2
                kgFloreodos = kg_por_arbol_floreo / 2
                NBinesBarrer = len(d_barrer)
                NBinesFloreo = len(d_floreo)

                # Extraer posiciones de bines
                H_H_1 = [row[1] for group in d_barrer.values() for row in group if len(row) >= 3 and row[2] == "bin"]
                H_H_2 = [row[1] for group in d_barrer_2.values() for row in group if len(row) >= 3 and row[2] == "bin"]
                H_H_3 = [row[1] for group in d_barrer_3.values() for row in group if len(row) >= 3 and row[2] == "bin"]
                H_H_4 = [row[1] for group in d_barrer_4.values() for row in group if len(row) >= 3 and row[2] == "bin"]
                H_H_F = [row[1] for group in d_floreo.values() for row in group if len(row) >= 3 and row[2] == "bin"]

                # Guardar en session state
                st.session_state['resultados'] = {
                    'arboles_por_hilera': arboles_por_hilera,
                    'sep_hileras_m': sep_hileras_m,
                    'sep_arboles_m': sep_arboles_m,
                    'kgBarrer': kg_por_arbol_barrer,
                    'kgFloreo': kg_por_arbol_floreo,
                    'kgBarrerdos': kgBarrerdos,
                    'kgFloreodos': kgFloreodos,
                    'NBinesBarrer': NBinesBarrer,
                    'NBinesFloreo': NBinesFloreo,
                    'H_H_1': H_H_1,
                    'H_H_2': H_H_2,
                    'H_H_3': H_H_3,
                    'H_H_4': H_H_4,
                    'H_H_F': H_H_F,
                    'txt_barrer': txt_barrer,
                    'txt_floreo': txt_floreo,
                    'txt_2_B': txt_2_B,
                    'txt_3_B': txt_3_B,
                    'txt_4_B': txt_4_B,
                    'd_barrer': d_barrer,
                    'd_floreo': d_floreo,
                    'campo': campo,
                    'empresa': empresa,
                    'fecha': fecha
                }

                st.success("✅ Optimización generada exitosamente!")

        except Exception as e:
            st.error(f"❌ Error al procesar: {str(e)}")
            st.error("Verifica que la lista de árboles tenga el formato correcto: [60,60,60,60]")

    # Mostrar tabla de datos si se cargó Excel
    if uploaded_file is not None and df_cosecha is not None:
        with st.expander("📊 Ver datos del plan de cosecha completo"):
            st.dataframe(df_cosecha, use_container_width=True)

    # Mostrar resultados si existen
    if 'resultados' in st.session_state:
        res = st.session_state['resultados']

        # Tabs para organizar información
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Resumen", "🌲 Floreo", "🍎 Barrer", "📄 Archivos Generados"])

        with tab1:
            st.header("Resumen de Optimización")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Campo", res['campo'])
                st.metric("Empresa", res['empresa'])
                st.metric("Fecha", res['fecha'])

            with col2:
                st.metric("Hileras", len(res['arboles_por_hilera']))
                st.metric("Total Árboles", sum(res['arboles_por_hilera']))
                st.metric("Sep. Hileras (m)", res['sep_hileras_m'])

            with col3:
                st.metric("Bines Floreo", res['NBinesFloreo'])
                st.metric("Bines Barrer", res['NBinesBarrer'])
                st.metric("Sep. Árboles (m)", res['sep_arboles_m'])

            st.subheader("📏 Pasillos Horizontales Óptimos")

            pasillo_col1, pasillo_col2 = st.columns(2)

            with pasillo_col1:
                st.markdown("**Configuración con 1 bin por posición:**")
                if len(res['H_H_1']) >= 2:
                    st.write(f"- Cada {res['H_H_1'][1] - res['H_H_1'][0]:.1f} metros")
                    st.write(f"- Primer pasillo en el metro {res['H_H_1'][0]:.1f}")

                st.markdown("**Configuración con 2 bines por posición:**")
                if len(res['H_H_2']) >= 2:
                    st.write(f"- Cada {res['H_H_2'][1] - res['H_H_2'][0]:.1f} metros")
                    st.write(f"- Primer pasillo en el metro {res['H_H_2'][0]:.1f}")

            with pasillo_col2:
                st.markdown("**Configuración con 3 bines por posición:**")
                if len(res['H_H_3']) >= 2:
                    st.write(f"- Cada {res['H_H_3'][1] - res['H_H_3'][0]:.1f} metros")
                    st.write(f"- Primer pasillo en el metro {res['H_H_3'][0]:.1f}")

                st.markdown("**Configuración con 4 bines por posición:**")
                if len(res['H_H_4']) >= 2:
                    st.write(f"- Cada {res['H_H_4'][1] - res['H_H_4'][0]:.1f} metros")
                    st.write(f"- Primer pasillo en el metro {res['H_H_4'][0]:.1f}")

        with tab2:
            st.header("🌲 Cosecha en Floreo")

            st.markdown(f"""
            **Información:**
            - Se estima **{res['kgFloreo']} kg** de fruta por planta en floreo ({res['kgFloreodos']} kg por cara)
            - Se recomienda colocar **{res['NBinesFloreo']} bines** equidistantes

            **Indicaciones:**
            - Considerar 4 hileras de árboles
            - Colocar 2-3 personas por cada cara
            - En la calle de en medio colocar los bines
            - Ubicar bines cerca de los pasillos horizontales
            """)

            # Generar gráfico floreo
            if len(res['H_H_F']) >= 2:
                fig_floreo = plot_hileras_arboles_y_pasillos(
                    arboles_por_hilera=res['arboles_por_hilera'],
                    sep_hileras_m=res['sep_hileras_m'],
                    sep_arboles_m=res['sep_arboles_m'],
                    HHOunB=res['H_H_F'][1] - res['H_H_F'][0],
                    primerHunB=res['H_H_F'][0],
                    kgBarrer=res['kgFloreo'],
                    kgBarrerdos=res['kgFloreodos'],
                    num_clases=res['NBinesFloreo'],
                    titulo="Disposición Óptima - Floreo",
                    bins_y=res['H_H_F'],
                    draw_aisles=True,
                    is_floreo=True
                )
                st.pyplot(fig_floreo)

        with tab3:
            st.header("🍎 Cosecha en Barrer")

            st.markdown(f"""
            **Información:**
            - Se estima **{res['kgBarrer']} kg** de fruta por planta en barrer ({res['kgBarrerdos']} kg por cara)
            - Se recomienda colocar **{res['NBinesBarrer']} bines** equidistantes

            **Indicaciones:**
            - Considerar 4 hileras de árboles
            - Colocar 2-3 personas por cada cara
            - En la calle de en medio colocar los bines
            - Ubicar bines cerca de los pasillos horizontales
            """)

            # Generar gráfico barrer
            if len(res['H_H_1']) >= 2:
                fig_barrer = plot_hileras_arboles_y_pasillos(
                    arboles_por_hilera=res['arboles_por_hilera'],
                    sep_hileras_m=res['sep_hileras_m'],
                    sep_arboles_m=res['sep_arboles_m'],
                    HHOunB=res['H_H_1'][1] - res['H_H_1'][0],
                    primerHunB=res['H_H_1'][0],
                    kgBarrer=res['kgBarrer'],
                    kgBarrerdos=res['kgBarrerdos'],
                    num_clases=res['NBinesBarrer'],
                    titulo="Disposición Óptima - Barrer",
                    bins_y=res['H_H_1'],
                    draw_aisles=True,
                    is_floreo=False
                )
                st.pyplot(fig_barrer)

            # Gráfico de pasillos horizontales
            st.subheader("Pasillos Horizontales (1 bin/posición)")
            if len(res['H_H_1']) >= 2:
                fig_pasillos = plot_hileras_arboles_y_pasillos(
                    arboles_por_hilera=res['arboles_por_hilera'],
                    sep_hileras_m=res['sep_hileras_m'],
                    sep_arboles_m=res['sep_arboles_m'],
                    HHOunB=res['H_H_1'][1] - res['H_H_1'][0],
                    primerHunB=res['H_H_1'][0],
                    kgBarrer=res['kgBarrer'],
                    kgBarrerdos=res['kgBarrerdos'],
                    num_clases=6,
                    titulo="Pasillos Horizontales Óptimos",
                    bins_y=res['H_H_1'],
                    draw_aisles=True
                )
                st.pyplot(fig_pasillos)

        with tab4:
            st.header("📄 Archivos Generados")

            st.subheader("Descargar Mapas TXT")

            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    label="📥 Descargar Mapa Floreo (.txt)",
                    data=res['txt_floreo'],
                    file_name="mapa_floreo.txt",
                    mime="text/plain"
                )

                st.download_button(
                    label="📥 Descargar Mapa Barrer (.txt)",
                    data=res['txt_barrer'],
                    file_name="mapa_barrer.txt",
                    mime="text/plain"
                )

            with col2:
                st.download_button(
                    label="📥 Descargar Mapa 2 Bines (.txt)",
                    data=res['txt_2_B'],
                    file_name="mapa_2_bines.txt",
                    mime="text/plain"
                )

                st.download_button(
                    label="📥 Descargar Mapa 3 Bines (.txt)",
                    data=res['txt_3_B'],
                    file_name="mapa_3_bines.txt",
                    mime="text/plain"
                )

            st.download_button(
                label="📥 Descargar Mapa 4 Bines (.txt)",
                data=res['txt_4_B'],
                file_name="mapa_4_bines.txt",
                mime="text/plain"
            )

            # Vista previa de archivos
            st.subheader("Vista Previa de Mapas")

            preview_option = st.selectbox(
                "Seleccionar mapa para vista previa:",
                ["Floreo", "Barrer", "2 Bines", "3 Bines", "4 Bines"]
            )

            preview_map = {
                "Floreo": res['txt_floreo'],
                "Barrer": res['txt_barrer'],
                "2 Bines": res['txt_2_B'],
                "3 Bines": res['txt_3_B'],
                "4 Bines": res['txt_4_B']
            }

            with st.expander("Ver contenido del mapa"):
                st.code(preview_map[preview_option], language="python")

    else:
        st.info("👈 Puedes cargar un archivo Excel del plan de cosecha o configurar los parámetros manualmente en el panel lateral")

        # Mostrar información de ayuda
        st.markdown("""
        ### 🔍 Acerca de esta herramienta

        Esta aplicación permite optimizar la ubicación de bines en campos de cosecha mediante:

        - **📂 Carga automática desde Excel**: Importa el plan de cosecha y automáticamente calcula parámetros
        - **Análisis de estructura del huerto**: Define hileras, árboles y separaciones
        - **Optimización de ubicación de bines**: Calcula posiciones óptimas para minimizar distancias
        - **Cálculo de pasillos horizontales**: Determina dónde abrir pasillos para mayor eficiencia
        - **Generación de visualizaciones**: Gráficos claros de la disposición óptima
        - **Exportación de datos**: Descarga mapas en formato txt para uso en campo

        #### 📝 Opciones de uso:

        **Opción 1: Cargar Excel** (Recomendado)
        1. Sube tu archivo Excel del plan de cosecha
        2. Selecciona el CECO y cuartel deseado
        3. Los parámetros se calcularán automáticamente
        4. Presiona "Generar Optimización"

        **Opción 2: Configuración manual**
        1. Ingresa manualmente los parámetros en el panel lateral
        2. Define árboles por hilera como lista: [60,60,60,60]
        3. Ajusta kg por árbol según tipo de cosecha
        4. Presiona "Generar Optimización"

        #### 🔑 Parámetros clave:
        - **Árboles por hilera**: Lista de árboles en cada hilera
        - **Kg por árbol**: Estimación de fruta por árbol según tipo de cosecha
        - **Capacidad de bin**: Peso máximo que puede contener cada bin
        - **Caras extra**: Ajuste para "cuadrar" clases de cosecha
        """)


if __name__ == "__main__":
    main()