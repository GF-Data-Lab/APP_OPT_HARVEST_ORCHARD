import matplotlib.pyplot as plt
from string import Template
from pathlib import Path
import ast
from math import ceil
from typing import Dict, List, Tuple

Punto = Tuple[float, float, str, int]  # (x, y, tipo, id)


def generar_mapa_cuadrado_txt(
    arboles_por_hilera: List[int],
    capacidad_bin_kg: float,
    kg_por_arbol: float,
    sep_hileras_m: float = 4.0,
    sep_arboles_m: float = 2.0,
    extra_caras: int = 2,
    output_path: str = "mapa_por_capacidad_cuadrado.txt",
) -> str:
    """Construye el mapa y lo guarda como .txt (claves numéricas sin comillas).
    Devuelve la ruta de salida.
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
    caras_min = ceil(capacidad_bin_kg / kg_por_cara)  # mínimo de caras para llenar
    caras_por_clase = caras_min + extra_caras  # "cuadrado" (ej: +2 => 32)

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
            items.append((x, y, "cara", cara_id_global))
            ys.append(y)
            cara_id_global += 1
            idx += 1

        # Bin centrado en Y dentro de la clase
        y_bin = (min(ys) + max(ys)) / 2.0 if ys else 0.0
        items.append((x_bin, y_bin, "bin", 1))

        clases[clase_idx] = items
        clase_idx += 1

    if not clases:
        clases[1] = [(x_bin, 0.0, "bin", 1)]

    # Guardar como .txt con claves numéricas (sin comillas)
    def dict_to_text(d: Dict[int, List[Punto]]) -> str:
        lines = ["{"]
        for k in sorted(d.keys()):
            lines.append(f"  {k}: [")
            for x, y, tipo, idx_item in d[k]:
                lines.append(f'    [{x}, {y}, "{tipo}", {idx_item}],')
            lines.append("  ],")
        lines.append("}")
        return "\n".join(lines)

    txt = dict_to_text(clases)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(txt)

    return output_path, txt


# -----------------------------
# CLI simple por input()
# -----------------------------
if __name__ == "__main__":
    print("Ingrese el mapa global (lista). Ejemplo: [60,60,60,60]")
    lista_str = input("> ").strip()
    try:
        arboles_por_hilera = ast.literal_eval(lista_str)
        assert isinstance(arboles_por_hilera, list) and all(
            isinstance(x, int) and x >= 0 for x in arboles_por_hilera
        )
    except Exception as e:
        raise SystemExit(f"Formato inválido para la lista: {e}")

    print("Capacidad del bin (kg). Ejemplo: 300")
    capacidad_bin_kg = float(input("> ").strip())

    print("Kg fijos por árbol en floreo. Ejemplo: 20")
    kg_por_arbol_floreo = float(input("> ").strip())

    print("Kg fijos por árbol en barrer. Ejemplo: 20")
    kg_por_arbol = float(input("> ").strip())

    # Parámetros opcionales (puedes presionar Enter para usar los defaults)
    def read_float(prompt: str, default: float) -> float:
        s = input(prompt).strip()
        return float(s) if s else default

    sep_hileras_m = read_float("Separación entre hileras en metros [default 4.0]: ", 4.0)
    sep_arboles_m = read_float("Separación entre árboles en metros [default 2.0]: ", 2.0)

    # Por defecto, sumamos +2 caras para “cuadrar” (como acordado)
    s_extra = input("Caras extra para 'cuadrar' por clase (ej. 2) [default 2]: ").strip()
    extra_caras = int(s_extra) if s_extra else 2

    s_extra_floreo = input(
        "Caras extra para 'cuadrar' por clase floreo (ej. 2) [default 2]: ",
    ).strip()
    extra_caras_floreo = int(s_extra_floreo) if s_extra_floreo else 2

    out = input("Ruta de salida .txt [default mapa_por_capacidad_cuadrado.txt]: ").strip()
    output_path = out if out else "mapa_por_capacidad_cuadrado.txt"

    path, txt_floreo = generar_mapa_cuadrado_txt(
        arboles_por_hilera=arboles_por_hilera,
        capacidad_bin_kg=capacidad_bin_kg,
        kg_por_arbol=kg_por_arbol_floreo,
        sep_hileras_m=sep_hileras_m,
        sep_arboles_m=sep_arboles_m,
        extra_caras=extra_caras_floreo,
        output_path=output_path,
    )
    path, txt_barrer = generar_mapa_cuadrado_txt(
        arboles_por_hilera=arboles_por_hilera,
        capacidad_bin_kg=capacidad_bin_kg,
        kg_por_arbol=kg_por_arbol,
        sep_hileras_m=sep_hileras_m,
        sep_arboles_m=sep_arboles_m,
        extra_caras=extra_caras,
        output_path=output_path,
    )

    path, txt_2_B = generar_mapa_cuadrado_txt(
        arboles_por_hilera=arboles_por_hilera,
        capacidad_bin_kg=capacidad_bin_kg * 2,
        kg_por_arbol=kg_por_arbol,
        sep_hileras_m=sep_hileras_m,
        sep_arboles_m=sep_arboles_m,
        extra_caras=extra_caras * 2,
        output_path=output_path,
    )
    path, txt_3_B = generar_mapa_cuadrado_txt(
        arboles_por_hilera=arboles_por_hilera,
        capacidad_bin_kg=capacidad_bin_kg * 3,
        kg_por_arbol=kg_por_arbol,
        sep_hileras_m=sep_hileras_m,
        sep_arboles_m=sep_arboles_m,
        extra_caras=extra_caras * 3,
        output_path=output_path,
    )
    path, txt_4_B = generar_mapa_cuadrado_txt(
        arboles_por_hilera=arboles_por_hilera,
        capacidad_bin_kg=capacidad_bin_kg * 4,
        kg_por_arbol=kg_por_arbol,
        sep_hileras_m=sep_hileras_m,
        sep_arboles_m=sep_arboles_m,
        extra_caras=extra_caras * 4,
        output_path=output_path,
    )

    print(f"\n✅ Mapa generado y guardado en: {path}")
    print("Listo.")


plantilla = Template(
    r"""% !TEX TS-program = pdflatex
\documentclass[final]{beamer}

% ========= Paleta roja (ajustable) =========
\definecolor{RPrim}{RGB}{178,34,34}
\definecolor{RSec}{RGB}{220,20,60}
\definecolor{RSoft}{RGB}{255,235,238}
\definecolor{RDeep}{RGB}{128,0,0}
\definecolor{RGray}{RGB}{80,80,80}
\definecolor{Sand}{RGB}{247,246,245}

% ========= Estilo general =========
\setbeamercolor{title}{fg=white,bg=RPrim}
\setbeamercolor{frametitle}{fg=white,bg=RSec}
\setbeamercolor{normal text}{fg=black,bg=Sand}
\setbeamercolor{block title}{fg=white,bg=RPrim}
\setbeamercolor{block body}{fg=black,bg=RSoft}
\setbeamertemplate{itemize items}[circle]

% ========= Datos del proyecto (parametrizados) =========
\newcommand{\Campo}{$Campo}
\newcommand{\Empresa}{$Empresa}
\newcommand{\Proyecto}{$Proyecto}
\newcommand{\Fecha}{$Fecha}
\newcommand{\kgBarrer}{$kgBarrer}
\newcommand{\kgFloreo}{$kgFloreo}
\newcommand{\kgBarrerdos}{$kgBarrerdos}
\newcommand{\kgFloreodos}{$kgFloreodos}
\newcommand{\NBinesFloreo}{$NBinesFloreo}
\newcommand{\NBinesBarrer}{$NBinesBarrer}
\newcommand{\HHOunB}{$HHOunB}
\newcommand{\HHOdosB}{$HHOdosB}
\newcommand{\HHOtresB}{$HHOtresB}
\newcommand{\HHOcuaB}{$HHOcuaB}
\newcommand{\primerHunB}{$primerHunB}
\newcommand{\primerHdosB}{$primerHdosB}
\newcommand{\primerHtresB}{$primerHtresB}
\newcommand{\primerHcuaB}{$primerHcuaB}



% ========= Paquetes base =========
\usepackage[spanish,es-nodecimaldot]{babel}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[size=a2,orientation=portrait,scale=1.12]{beamerposter}
\usefonttheme{professionalfonts}
\usepackage{lmodern}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{multicol}
\usepackage{wrapfig}
\usepackage{tikz}
\usetikzlibrary{positioning,shapes,arrows.meta,calc,fit}
\usepackage{tcolorbox}
\tcbuselibrary{skins,breakable}

% ========= Utilidades visuales =========
\newtcolorbox{GBlock}[2][]{enhanced,breakable,
  colback=RSoft,colframe=RPrim!85!black,
  coltitle=white,fonttitle=\bfseries\large,
  title=#2,boxrule=1.2pt,rounded corners=2mm,
  attach boxed title to top left={yshift=-2mm,xshift=2mm},
  boxed title style={size=small,colback=RPrim,rounded corners=2mm},
  #1}
\newtcolorbox{SBlock}[2][]{enhanced,breakable,
  colback=Sand,colframe=RSec!80!black,
  coltitle=white,fonttitle=\bfseries,
  title=#2,boxrule=0.9pt,rounded corners=2mm,
  attach boxed title to top left={yshift=-2mm,xshift=2mm},
  boxed title style={size=small,colback=RSec,rounded corners=2mm},
  #1}

\newcommand{\imgplaceholder}[2]{%
  \begin{center}
    \fbox{\rule{0pt}{#2}\rule{#1}{0pt}}
    \par\vspace{2mm}\footnotesize\color{RGray}
  \end{center}
}

% ========= Documento =========
\title{\Proyecto{} — \Campo}
\author{\Empresa}
\date{\Fecha}

\begin{document}
\begin{frame}[t]

% ====== Cabecera con logo + título ======
\begin{columns}[t,totalwidth=\textwidth]
  \begin{column}{0.72\textwidth}
    \begin{tcolorbox}[colback=RPrim,colframe=RPrim,sharp corners,boxrule=0pt,rounded corners=2mm]
      {\color{white}\LARGE \textbf{\Proyecto{}}}\\[2mm]
      {\color{white}\Large \textbf{Campo: \Campo}}\\[1mm]
      {\color{white}\normalsize \Fecha}
    \end{tcolorbox}
  \end{column}
  \begin{column}{0.28\textwidth}
    \begin{tcolorbox}[colback=Sand,colframe=Sand,boxrule=0pt,rounded corners=2mm]
      \centering
      \includegraphics[width=0.9\linewidth]{logo.png}
    \end{tcolorbox}
  \end{column}
\end{columns}

\vspace{4mm}

% ====== Fila 1: Resumen + Mapa satelital ======
\begin{columns}[t,totalwidth=\textwidth]
  \begin{column}{0.44\textwidth}
    \begin{GBlock}{¿De qué trata el proyecto?}
      \textbf{Objetivo:} Optimizar la cosecha en \Campo{}  reduciendo significativamente la distancia que recorren los cosecheros ubicando de manera óptima los bines, transformando la energía gastada en caminar en más totes cosechados por persona. \\[2mm]
      \textbf{Para realizar el modelo se requiere información del campo y el tipo de cosecha tal como:}
      \begin{itemize}
       \item Cantidad de hileras.
       \item Estimado de fruta en kg para cada tipo de cosecha (barrer, floreo y en temporada).
       \item Cantidad de arboles por hilera.
       \item Distancia entre arboles y calles.
       \item Hileras polinizantes.
       \item Pasillos horizontales.
      \end{itemize}
      \textbf{Qué entregamos:}
      \begin{itemize}
        \item Donde deberían ir ubicados los bines de manera óptima para cada tipo de cosecha.
        \item Cuantos trabajadores deberían estar en cada cara de hilera para cada tipo de cosecha.
      \end{itemize}
    \end{GBlock}

    \begin{SBlock}{Como lo logramos:}
     \textbf{ Se realiza un modelo computacional y matemático, donde agentes virtuales cosechan en un campo virtual con las mismas características del campo real, iterando en diferentes entornos y cosechando de diferentes maneras, se logra evidenciar la manera más óptima que reduce la distancia global e indivual por persona. }
    \end{SBlock}
  \end{column}

  \begin{column}{0.56\textwidth}
    \begin{GBlock}{Mapa satelital: \Campo}
    \includegraphics[width=\linewidth]{lihueimo 10784 C 1.png}
    \end{GBlock}
  \end{column}
\end{columns}
\vspace{3mm}

% ====== Fila 2: Escenarios (3 alternativas) ======
\begin{GBlock}{Indicaciones donde colocar los bines de manera óptima}
  \begin{columns}[t,totalwidth=\textwidth]
    % ---- Escenario 1 ----
    \begin{column}{0.5\textwidth}
      \begin{SBlock}{Floreo}
        \textbf{Información:}
        \begin{itemize}
          \item Se estima \kgFloreo kg de fruta por planta en floreo. (\kgFloreodos kg por cara)
          \item A continuación se muestra donde ubicar los bines suponiendo que en cada ubicación hay un solo bin.
        \end{itemize}
        \textbf{Indicaciones:}
        \begin{itemize}
          \item Considerar 4 hileras de arboles
          \item Colocar 2-3 personas por cada cara
          \item En la calle de en medio colocar los bines
          \item Ubicación de bines varía en la práctica, se recomienda colocar \NBinesFloreo bines equisdistantes en lo posible cerca de los pasillos horizontales.
        \end{itemize}
         \includegraphics[width=0.6\linewidth]{floreo (1).png}
      \end{SBlock}
    \end{column}

    % ---- Escenario 2 ----
    \begin{column}{0.5\textwidth}
     \begin{SBlock}{Barrer}
        \textbf{Información:}
        \begin{itemize}
          \item Se estima \kgBarrer kg de fruta por planta en barrer. (\kgBarrerdos kg por cara)
          \item A continuación se muestra donde ubicar los bines suponiendo que en cada ubicación hay un solo bin.
        \end{itemize}
        \textbf{Indicaciones:}
        \begin{itemize}
          \item Considerar 4 hileras de arboles
          \item Colocar 2-3 personas por cada cara
          \item En la calle de en medio colocar los bines
          \item Ubicación de bines varía en la práctica, se recomienda colocar \NBinesBarrer bines equisdistantes en lo posible cerca de los pasillos horizontales.
        \end{itemize}
         \includegraphics[width=0.618\linewidth]{barrer.png}
      \end{SBlock}
    \end{column}


  \end{columns}
\end{GBlock}

\vspace{3mm}

% ====== Fila 3: Ubicación de hileras horizontales ======
\begin{GBlock}{Definición de pasillos horizontales óptimos}
  \begin{columns}[t,totalwidth=\textwidth]
    \begin{column}{0.54\textwidth}
      \begin{SBlock}{Indicaciones pasillos horizontales según modelo}
        \begin{itemize}
          \item Dependiendo de cuantos bines se desean en cada posición la cantidad de hileras horizontales óptimas varía.
          \item Se recomienda siempre dejar los bines cerca de cada intersección.
          \item Con un bin por posición, el modelo arroja que se tiene que hacer pasillos horizontales cada \textbf{\HHOunB} metros. Colocando el primer pasillo en el metro \primerHunB.
           \item Con dos bines por posición, cada \textbf{\HHOdosB}. Colocando el primer pasillo en el metro \primerHdosB.
          \item Con tres bines por posición, cada \textbf{\HHOtresB}. Colocando el primer pasillo en el metro \primerHtresB.
          \item Con cuatro bines por posición, cada \textbf{\HHOcuaB}. Colocando el primer pasillo en el metro \primerHcuaB.
          \item \textbf{Mientras mas pasillos horizontales existan, menos distancia recorrerán los cosecheros globalmente, sin embargo existe una perdida de fruta por cada pasillo que se abra}
          \item El modelo arroja los resultados según el parámetro de estimación de fruta de cada árbol que son \kgBarrer kg de fruta por árbol, es decir, \kgBarrerdos kg por cara, y asegura el que todos los bines en cada posición se llenen en promedio

        \end{itemize}

      \end{SBlock}


    \end{column}

    \begin{column}{0.46\textwidth}
      \begin{SBlock}{Gráfico explicativo pasillos}
        \includegraphics[width=0.6\linewidth]{pasillos horizontales.png}
      \end{SBlock}
    \end{column}
  \end{columns}
\end{GBlock}
\end{frame}
\end{document}
"""
)

d_barrer = ast.literal_eval(txt_barrer)
d_floreo = ast.literal_eval(txt_floreo)

d_barrer_2 = ast.literal_eval(txt_2_B)
d_barrer_3 = ast.literal_eval(txt_3_B)
d_barrer_4 = ast.literal_eval(txt_4_B)

kgBarrer = 11.2
kgBarrerdos = kgBarrer / 2
kgFloreo = 2.8
kgFloreodos = kgFloreo / 2
NBinesBarrer = len(d_barrer)
NBinesFloreo = len(d_floreo)


H_H_1 = [
    row[1] for group in d_barrer.values() for row in group if len(row) >= 3 and row[2] == "bin"
]
H_H_2 = [
    row[1] for group in d_barrer_2.values() for row in group if len(row) >= 3 and row[2] == "bin"
]
H_H_3 = [
    row[1] for group in d_barrer_3.values() for row in group if len(row) >= 3 and row[2] == "bin"
]
H_H_4 = [
    row[1] for group in d_barrer_4.values() for row in group if len(row) >= 3 and row[2] == "bin"
]

H_H_F = [
    row[1] for group in d_floreo.values() for row in group if len(row) >= 3 and row[2] == "bin"
]


# Valores a inyectar
data = {
    "Campo": "Lihueimo CECO 10784 Cuartel 1",
    "Empresa": "Garcés",
    "Proyecto": "Optimización de Cosecha y Logística en Campos",
    "Fecha": "04/11/2025",
    "kgBarrer": kgBarrer,
    "kgFloreo": kgFloreo,
    "kgBarrerdos": kgBarrerdos,
    "kgFloreodos": kgFloreodos,
    "NBinesFloreo": NBinesFloreo,
    "NBinesBarrer": NBinesBarrer,
    "HHOunB": H_H_1[1] - H_H_1[0],
    "HHOdosB": H_H_2[1] - H_H_2[0],
    "HHOtresB": H_H_3[1] - H_H_3[0],
    "HHOcuaB": H_H_4[1] - H_H_4[0],
    "primerHunB": H_H_1[0],
    "primerHdosB": H_H_2[0],
    "primerHtresB": H_H_3[0],
    "primerHcuaB": H_H_4[0],
}


tex = plantilla.substitute(**data)

# Guardar a archivo .tex
Path("LATEX.tex").write_text(tex, encoding="utf-8")
print("Archivo generado: poster_parametrizado.tex")


def plot_hileras_arboles_y_pasillos(
    arboles_por_hilera,
    sep_hileras_m,
    sep_arboles_m,
    HHOunB,
    primerHunB,
    kgBarrer,
    kgBarrerdos,
    num_clases=4,
    margen_x=1.0,
    out_path=None,
    dpi=180,
    show=False,
    titulo="Hileras, árboles y pasillos horizontales (1 bin/posición)",
    subtitulo=None,
    caption=None,
    # aisles toggle
    draw_aisles=True,
    # --- BINS CONFIG ---
    bins_y=None,  # e.g., [5.0, 12.0, 21.0]
    bins_xy=None,  # e.g., [(6.0, 11.0), (6.0, 23.0)]
    bin_marker="o",
    bin_size=40,
):
    n_hileras = len(arboles_por_hilera)
    if n_hileras == 0:
        raise ValueError("Debe haber al menos 1 hilera.")

    # X centers
    centros_x = [i * sep_hileras_m for i in range(n_hileras)]
    etiquetas_x = [f"Hilera {i + 1}" for i in range(n_hileras)]

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
        ax.vlines(cx, y_min, y_max, linewidth=1, linestyles="dashed")
        n_arboles = arboles_por_hilera[i]
        ys = [j * sep_arboles_m for j in range(n_arboles) if j * sep_arboles_m <= y_max]
        xs = [cx] * len(ys)
        ax.scatter(xs, ys, s=14)

    # Dibujar pasillos
    x0 = min(centros_x) - margen_x
    x1 = max(centros_x) + margen_x
    Y = []
    for idx, y in enumerate(y_pasillos, start=1):
        ax.hlines(y, x0, x1, linewidth=2)
        ax.annotate(
            f"Pasillo {idx}, {y} m",
            xy=(x1, y),
            xytext=(5, 2),
            textcoords="offset points",
            ha="left",
            va="bottom",
        )
        Y.append(y)

    # --- Dibujar BINes según lista ---
    if out_path != "floreo (1).png":
        bins_y = Y

    if bins_xy:
        xs, ys = zip(*bins_xy) if len(bins_xy) else ([], [])
        if xs:
            ax.scatter(xs, ys, s=bin_size, marker=bin_marker, zorder=5)
    elif bins_y:
        if n_hileras >= 3:
            x_bin = (centros_x[1] + centros_x[2]) / 2.0
        elif n_hileras == 2:
            x_bin = (centros_x[0] + centros_x[1]) / 2.0
        else:
            x_bin = centros_x[0] + sep_hileras_m / 2.0
        xs = [x_bin] * len(bins_y)
        ax.scatter(xs, bins_y, s=bin_size, marker=bin_marker, zorder=5)
        ax.annotate(
            "bines",
            xy=(x_bin, bins_y[0] if bins_y else 0.0),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Ejes
    ax.set_xticks(centros_x, etiquetas_x, rotation=0)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Y (m)")

    # Títulos
    if subtitulo is None:
        subtitulo = f"kg por árbol (barrer): {kgBarrer} | kg por cara: {kgBarrerdos}"
    fig.suptitle(titulo, fontsize=14, y=0.98)
    ax.set_title(subtitulo, fontsize=10, pad=8)

    # Caption
    if caption is None:
        caption = "Bines según lista de posiciones."
    fig.text(0.995, 0.01, caption, ha="right", va="bottom", fontsize=9)

    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.93])

    if out_path is not None:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


# Demo 1: bins_y (misma X entre hilera 2 y 3)

plot_hileras_arboles_y_pasillos(
    arboles_por_hilera=arboles_por_hilera,
    sep_hileras_m=sep_hileras_m,
    sep_arboles_m=sep_arboles_m,
    HHOunB=H_H_1[1] - H_H_1[0],
    primerHunB=H_H_1[0],
    kgBarrer=kgBarrer,
    kgBarrerdos=kgBarrerdos,
    num_clases=6,
    out_path="pasillos horizontales.png",
    show=False,
    bins_y=H_H_1,
    draw_aisles=True,
)

plot_hileras_arboles_y_pasillos(
    arboles_por_hilera=arboles_por_hilera,
    sep_hileras_m=sep_hileras_m,
    sep_arboles_m=sep_arboles_m,
    HHOunB=H_H_1[1] - H_H_1[0],
    primerHunB=H_H_1[0],
    kgBarrer=kgBarrer,
    kgBarrerdos=kgBarrerdos,
    num_clases=NBinesBarrer,
    out_path="Barrer.png",
    show=False,
    bins_y=H_H_1,
    draw_aisles=True,
)

plot_hileras_arboles_y_pasillos(
    arboles_por_hilera=arboles_por_hilera,
    sep_hileras_m=sep_hileras_m,
    sep_arboles_m=sep_arboles_m,
    HHOunB=H_H_1[1] - H_H_1[0],
    primerHunB=H_H_1[0],
    kgBarrer=kgBarrer,
    kgBarrerdos=kgBarrerdos,
    num_clases=NBinesBarrer,
    out_path="floreo (1).png",
    show=False,
    bins_y=H_H_F,
    draw_aisles=True,
)
print("Guardado en:", out)
