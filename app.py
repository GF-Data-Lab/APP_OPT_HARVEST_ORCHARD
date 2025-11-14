# -*- coding: utf-8 -*-
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="APP OPT HARVEST ORCHARD",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #558B2F;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #F1F8E9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🌳 APP OPT HARVEST ORCHARD</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Sistema de Optimización de Cosecha en Huertos</div>', unsafe_allow_html=True)

# Imagen o logo (si existe)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("---")

# Introducción
st.markdown("""
## 👋 Bienvenido al Sistema de Optimización de Cosecha

Esta aplicación web permite **optimizar la logística de cosecha en campos frutales** mediante algoritmos
avanzados de optimización y visualización interactiva.

### 🎯 Objetivo Principal

Reducir significativamente la distancia que recorren los cosecheros mediante la **ubicación óptima de bines**,
transformando la energía gastada en caminar en más totes cosechados por persona.
""")

st.markdown("---")

# Características principales
st.markdown("## ✨ Características Principales")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-box">
        <h3>🎯 Optimización Automática</h3>
        <p>Cálculo de posiciones óptimas de bines basado en:</p>
        <ul>
            <li>Capacidad de bines</li>
            <li>Producción estimada por árbol</li>
            <li>Estructura del huerto</li>
            <li>Tipo de cosecha (floreo, barrer, temporada)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-box">
        <h3>📊 Visualización Interactiva</h3>
        <p>Gráficos claros y detallados de:</p>
        <ul>
            <li>Distribución de hileras y árboles</li>
            <li>Ubicación de bines</li>
            <li>Pasillos horizontales óptimos</li>
            <li>Configuraciones para múltiples escenarios</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
        <h3>📄 Generación de Documentos</h3>
        <p>Exportación de resultados en múltiples formatos:</p>
        <ul>
            <li>Mapas de ubicación (.txt)</li>
            <li>Documentos LaTeX para impresión</li>
            <li>Visualizaciones en alta resolución</li>
            <li>Reportes detallados</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-box">
        <h3>🗺️ Análisis de Parcelas</h3>
        <p>Herramientas avanzadas para:</p>
        <ul>
            <li>Importación de archivos KML</li>
            <li>Visualización geoespacial</li>
            <li>Cálculo de áreas y distancias</li>
            <li>Análisis de configuraciones</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Cómo usar
st.markdown("## 🚀 Cómo Usar la Aplicación")

st.markdown("""
<div class="info-box">
    <h4>1️⃣ Navegar por las páginas</h4>
    <p>Utiliza el menú lateral (👈) para acceder a los diferentes módulos de la aplicación.</p>
</div>

<div class="info-box">
    <h4>2️⃣ Configurar parámetros</h4>
    <p>Ingresa los datos de tu campo: hileras, árboles, separaciones y estimaciones de producción.</p>
</div>

<div class="info-box">
    <h4>3️⃣ Generar optimización</h4>
    <p>El sistema calculará automáticamente la configuración óptima para tu campo.</p>
</div>

<div class="info-box">
    <h4>4️⃣ Visualizar resultados</h4>
    <p>Explora los gráficos interactivos y las recomendaciones generadas.</p>
</div>

<div class="info-box">
    <h4>5️⃣ Exportar documentos</h4>
    <p>Descarga los mapas y configuraciones para uso en campo.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Módulos disponibles
st.markdown("## 📚 Módulos Disponibles")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📝 Automatización Lihueimo
    Sistema completo de optimización específico para el campo Lihueimo.
    Incluye generación de mapas, cálculo de pasillos y exportación de documentos.
    """)

with col2:
    st.markdown("""
    ### 🔧 Optimizaciones v2-v11
    Diferentes versiones del motor de optimización con características
    específicas y mejoras incrementales.
    """)

with col3:
    st.markdown("""
    ### 📊 Visualización
    Herramientas avanzadas de análisis visual, importación de KML
    y visualización geoespacial.
    """)

st.markdown("---")

# Información técnica
st.markdown("## 🛠️ Tecnologías Utilizadas")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    **Frontend**
    - Streamlit
    - Matplotlib
    - Plotly
    """)

with col2:
    st.markdown("""
    **Algoritmos**
    - K-means
    - K-medoids
    - Optimización
    """)

with col3:
    st.markdown("""
    **Procesamiento**
    - NumPy
    - Pandas
    - SciPy
    """)

with col4:
    st.markdown("""
    **Formatos**
    - KML/GeoJSON
    - LaTeX
    - TXT/JSON
    """)

st.markdown("---")

# Footer
st.markdown("""
## 📞 Información Adicional

Para más detalles sobre el proyecto, consulta el archivo `README.md` en el repositorio.

### 📈 Beneficios Esperados

- ⏱️ **Reducción de tiempos**: Menos distancia recorrida por los cosecheros
- 📦 **Mayor eficiencia**: Más totes cosechados por persona
- 💰 **Ahorro de costos**: Optimización de recursos humanos
- 📊 **Mejor planificación**: Decisiones basadas en datos

---

<div style="text-align: center; color: #666;">
    <p><b>Sistema desarrollado por Equipo Garcés</b></p>
    <p>Versión 1.0.0 | Noviembre 2025</p>
</div>
""", unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📌 Navegación Rápida")
    st.markdown("""
    Selecciona una página del menú superior para comenzar:

    - **Automatización Lihueimo**: Optimización completa
    - **Optimización v2-v11**: Diferentes versiones
    - **Visualización**: Análisis visual
    """)

    st.markdown("---")
    st.markdown("### ℹ️ Ayuda")
    st.markdown("""
    Si necesitas ayuda:
    1. Revisa el README.md
    2. Consulta la documentación
    3. Contacta al equipo de soporte
    """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; font-size: 0.8rem; color: #666;">
        Made with ❤️ using Streamlit
    </div>
    """, unsafe_allow_html=True)
