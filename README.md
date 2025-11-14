# 🌳 APP OPT HARVEST ORCHARD

Sistema de Optimización de Cosecha en Huertos - Aplicación web interactiva para optimizar la ubicación de bines y planificación de cosecha en campos frutales.

## 📋 Descripción

Esta aplicación permite optimizar la logística de cosecha en campos frutales mediante algoritmos de optimización y visualización interactiva. El sistema calcula la ubicación óptima de bines, determina pasillos horizontales estratégicos y estima recursos humanos necesarios para minimizar distancias recorridas y maximizar eficiencia.

## ✨ Características

- **Optimización Automática de Bines**: Cálculo de posiciones óptimas basado en capacidad y producción estimada
- **Visualización Interactiva**: Gráficos de hileras, árboles y ubicación de bines
- **Planificación de Pasillos**: Determinación óptima de pasillos horizontales según configuración
- **Múltiples Escenarios**: Soporte para floreo, barrer, y cosecha en temporada
- **Generación de Documentos**: Exportación de mapas y configuraciones en formato TXT y LaTeX
- **Análisis de Parcelas**: Importación y análisis de archivos KML
- **Interfaz Multi-página**: Navegación intuitiva entre diferentes módulos

## 🚀 Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/usuario/APP_OPT_HARVEST_ORCHARD.git
   cd APP_OPT_HARVEST_ORCHARD
   ```

2. **Crear entorno virtual (recomendado)**
   ```bash
   python -m venv venv

   # En Windows:
   venv\Scripts\activate

   # En Linux/Mac:
   source venv/bin/activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Uso

### Iniciar la aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

### Navegación

La aplicación cuenta con múltiples páginas accesibles desde el menú lateral:

1. **Inicio**: Página principal con descripción general
2. **Automatización Lihueimo**: Sistema completo de optimización y generación de documentos
3. **Optimización v2-v11**: Diferentes versiones del motor de optimización
4. **Visualización**: Herramientas de análisis visual de parcelas

### Flujo de Trabajo Típico

1. **Configurar parámetros del campo**:
   - Número de hileras
   - Árboles por hilera
   - Separación entre hileras y árboles
   - Estimación de kg por árbol

2. **Seleccionar tipo de cosecha**:
   - Floreo (baja producción)
   - Barrer (alta producción)
   - Temporada

3. **Generar optimización**:
   - El sistema calcula posiciones óptimas
   - Visualiza configuración recomendada
   - Genera archivos descargables

4. **Exportar resultados**:
   - Mapas en formato TXT
   - Documentos LaTeX para impresión
   - Visualizaciones en PNG

## 📁 Estructura del Proyecto

```
APP_OPT_HARVEST_ORCHARD/
│
├── app.py                      # Aplicación principal (página de inicio)
│
├── pages/                      # Páginas de la aplicación
│   ├── 1_Automatizacion_Lihueimo.py
│   ├── 2_Optimizacion_v2.py
│   ├── 3_Optimizacion_v3.py
│   ├── 4_Optimizacion_v4.py
│   ├── 5_Optimizacion_v5.py
│   ├── 6_Optimizacion_v6.py
│   ├── 7_Optimizacion_v11.py
│   └── 8_Visualizacion.py
│
├── files/                      # Archivos del proyecto
│   ├── data/                   # Datos de entrada
│   │   ├── equipo1.json
│   │   ├── equipo4.json
│   │   ├── equipo4.txt
│   │   └── parcela.kml
│   │
│   ├── scripts/                # Scripts auxiliares
│   │   ├── modelo_base.py
│   │   ├── orchard_kmedoids_capacity.py
│   │   ├── Posicion_bines_k_means.py
│   │   ├── Simulaciones_gonza.py
│   │   ├── kml.py
│   │   └── Automatizacion_documento_lihueimo.py.py
│   │
│   ├── notebooks/              # Jupyter notebooks
│   │   └── Optimizacion_bines_y_personas.ipynb
│   │
│   └── images/                 # Imágenes y recursos visuales
│
├── requirements.txt            # Dependencias del proyecto
├── README.md                   # Este archivo
└── .gitignore                  # Archivos ignorados por git
```

## 🛠️ Tecnologías Utilizadas

- **Streamlit**: Framework para aplicaciones web interactivas
- **Python 3.x**: Lenguaje de programación principal
- **Matplotlib**: Visualización de gráficos
- **NumPy**: Cálculos numéricos
- **Pandas**: Manipulación de datos (si aplica)
- **scikit-learn**: Algoritmos de clustering (K-means, K-medoids)

## 📊 Módulos Principales

### 1. Automatización Lihueimo
Sistema completo de optimización que incluye:
- Generación de mapas de ubicación de bines
- Cálculo de pasillos horizontales óptimos
- Visualización de configuraciones
- Exportación de documentos LaTeX

### 2. Scripts de Optimización
- **modelo_base.py**: Modelo base de optimización
- **orchard_kmedoids_capacity.py**: Clustering con K-medoids considerando capacidad
- **Posicion_bines_k_means.py**: Posicionamiento con K-means
- **Simulaciones_gonza.py**: Simulaciones de escenarios

### 3. Análisis de Parcelas
- Importación de archivos KML
- Visualización geoespacial
- Cálculo de áreas y distancias

## 🎯 Casos de Uso

### Caso 1: Planificación de Cosecha en Floreo
```python
# Configuración típica
arboles_por_hilera = [60, 60, 60, 60]
kg_por_arbol = 2.8
capacidad_bin = 300
separacion_hileras = 4.0
separacion_arboles = 2.0
```

### Caso 2: Cosecha en Barrer
```python
# Configuración típica
arboles_por_hilera = [60, 60, 60, 60]
kg_por_arbol = 11.2
capacidad_bin = 300
separacion_hileras = 4.0
separacion_arboles = 2.0
```

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 👥 Autores

- Equipo de Desarrollo Garcés

## 📧 Contacto

Para preguntas o sugerencias, por favor contactar a través de los issues del repositorio.

## 🙏 Agradecimientos

- A todos los colaboradores del proyecto
- Empresas y campos que han permitido validar el sistema
- Comunidad de Streamlit y Python

## 📈 Roadmap

- [ ] Integración con sistemas GPS en tiempo real
- [ ] Módulo de predicción con Machine Learning
- [ ] Exportación a formatos GIS (Shapefile, GeoJSON)
- [ ] Dashboard de monitoreo en tiempo real
- [ ] API REST para integración con otros sistemas
- [ ] Soporte multi-idioma
- [ ] Aplicación móvil companion

## 🔧 Troubleshooting

### Problema: Error al instalar dependencias
**Solución**: Asegúrate de tener pip actualizado: `pip install --upgrade pip`

### Problema: La aplicación no se abre en el navegador
**Solución**: Abre manualmente `http://localhost:8501` en tu navegador

### Problema: Errores de visualización
**Solución**: Verifica que matplotlib esté correctamente instalado

---

**Versión**: 1.0.0
**Última actualización**: Noviembre 2025
