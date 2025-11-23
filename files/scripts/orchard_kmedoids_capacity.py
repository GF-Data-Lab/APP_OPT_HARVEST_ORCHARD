#!/usr/bin/env python3
# orchard_kmedoids_capacity.py
#
# ▸ Coloca k bines (medoides) en un huerto:
#     – distancia “rodeando las hileras” (no se puede cruzar filas)
#     – cada bin debe recibir entre N_target – slack y N_target + slack árboles
# ▸ Si la capacidad total es insuficiente, se avisa y se detiene.
# ▸ Si es suficiente pero hay sobrantes después del reparto,
#   se asignan al bin más cercano (aunque ese bin sobre-pase max_cap)
#   para garantizar siempre una solución.

import argparse
import json
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

# ────────────────────────────────────────────────────────────────
# 1 · Parámetros por defecto (cambiables por CLI)
# ────────────────────────────────────────────────────────────────
PARAMS = dict(
    trees_per_face=[30, 30, 30],  # árboles por cara  (una entrada por hilera) -180 arboles en total
    dx_row=4.0,  # m entre hileras
    dy_tree=2.0,  # m entre árboles sucesivos
    row_width=1.0,  # ancho físico bloqueado de la hilera
    k_bins=5,  # número de bines
    N_target=32,
    slack=10,  # capacidad deseada ± holgura
    seed=1,
)

# ── CLI ─────────────────────────────────────────────────────────
p = argparse.ArgumentParser(description="Constrained k-medoids for orchards")
p.add_argument("--trees", type=str, help='Ej. "[40,45,50,55]"')
p.add_argument("--dx", type=float, help="dist hilera–hilera (m)")
p.add_argument("--dy", type=float, help="dist árbol–árbol (m)")
p.add_argument("--roww", type=float, help="ancho de la hilera (m)")
p.add_argument("-k", "--bins", type=int, help="nº de bines")
p.add_argument("-t", "--target", type=int, help="árboles objetivo por bin")
p.add_argument("-s", "--slack", type=int, help="holgura ± (árboles)")
p.add_argument("--seed", type=int, help="semilla aleatoria")
args = p.parse_args()
if args.trees:
    PARAMS["trees_per_face"] = json.loads(args.trees)
if args.dx:
    PARAMS["dx_row"] = args.dx
if args.dy:
    PARAMS["dy_tree"] = args.dy
if args.roww:
    PARAMS["row_width"] = args.roww
if args.bins:
    PARAMS["k_bins"] = args.bins
if args.target:
    PARAMS["N_target"] = args.target
if args.slack:
    PARAMS["slack"] = args.slack
if args.seed:
    PARAMS["seed"] = args.seed
np.random.seed(PARAMS["seed"])
random.seed(PARAMS["seed"])

# ────────────────────────────────────────────────────────────────
# 2 · Geometría del huerto
# ────────────────────────────────────────────────────────────────
rows = len(PARAMS["trees_per_face"])
dx, dy = PARAMS["dx_row"], PARAMS["dy_tree"]
w_row = PARAMS["row_width"]
half_w = w_row / 2

trees = []  # (x, y, row, side)
for r, n in enumerate(PARAMS["trees_per_face"]):
    xc = r * dx
    for side in (0, 1):  # 0 oeste, 1 este
        x = xc + (-half_w if side == 0 else half_w)
        for k in range(n):
            trees.append((x, k * dy, r, side))
trees = np.asarray(trees, float)
XY = trees[:, :2]
row_id = trees[:, 2].astype(int)
side = trees[:, 3].astype(int)
N_tot = len(trees)

ROW_TOP, ROW_BOT = (max(PARAMS["trees_per_face"]) - 1) * dy, 0.0


# ────────────────────────────────────────────────────────────────
# 3 · Distancia bloqueada (pre-calcular matriz)
# ────────────────────────────────────────────────────────────────
def blocked_dist(p, q, rp, sp, rq, sq):
    if rp == rq and sp == sq:
        return np.linalg.norm(p - q)
    d_down = (p[1] - ROW_BOT) + (q[1] - ROW_BOT) + abs(p[0] - q[0])
    d_up = (ROW_TOP - p[1]) + (ROW_TOP - q[1]) + abs(p[0] - q[0])
    return min(d_up, d_down)


print("⌛  Construyendo matriz de distancias …")
D = np.zeros((N_tot, N_tot))
for i in range(N_tot):
    for j in range(i + 1, N_tot):
        D[i, j] = D[j, i] = blocked_dist(XY[i], XY[j], row_id[i], side[i], row_id[j], side[j])

# ────────────────────────────────────────────────────────────────
# 4 · Comprobación de capacidad global
# ────────────────────────────────────────────────────────────────
k = PARAMS["k_bins"]
min_cap = PARAMS["N_target"] - PARAMS["slack"]
max_cap = PARAMS["N_target"] + PARAMS["slack"]

if k * max_cap < N_tot:
    sys.exit(
        f"❌  Capacidad insuficiente: {k}×{max_cap} < {N_tot} árboles. Aumenta k / target / slack.",
    )


# ────────────────────────────────────────────────────────────────
# 5 · Heurística k-medoids con capacidad
# ────────────────────────────────────────────────────────────────
def init_medoids():
    med = [np.random.randint(N_tot)]
    while len(med) < k:
        dist2 = np.min(
            [[np.linalg.norm(XY[i] - XY[m]) ** 2 for m in med] for i in range(N_tot)],
            axis=1,
        )
        probs = dist2 / dist2.sum()
        med.append(np.random.choice(N_tot, p=probs))
    return med


def assign(points, medoids):
    clusters = [[] for _ in range(k)]
    leftovers = []
    # ordenar por “dificultad” (gap best–2nd best)
    order = sorted(
        points,
        key=lambda i: np.sort([D[i, m] for m in medoids])[1]
        - np.sort([D[i, m] for m in medoids])[0],
        reverse=True,
    )
    for i in order:
        for c, m in sorted(enumerate(medoids), key=lambda t: D[i, t[1]]):
            if len(clusters[c]) < max_cap:
                clusters[c].append(i)
                break
        else:
            leftovers.append(i)
    # cumplir mínimo robando al cluster más grande
    for c in range(k):
        while len(clusters[c]) < min_cap and leftovers:
            donor = max(range(k), key=lambda x: len(clusters[x]))
            if len(clusters[donor]) <= min_cap:
                break
            victim = max(clusters[donor], key=lambda j: D[j, medoids[donor]])
            clusters[donor].remove(victim)
            clusters[c].append(victim)
    return clusters, leftovers


def update_medoids(cl, med):
    changed = False
    for c, pts in enumerate(cl):
        best = min(pts, key=lambda j: D[j, pts].sum())
        if best != med[c]:
            med[c] = best
            changed = True
    return changed


medoids = init_medoids()
for _ in range(12):
    clusters, leftovers = assign(range(N_tot), medoids)
    if not update_medoids(clusters, medoids):
        break

# Asignar sobrantes al bin más cercano (sobrepasando max_cap si es necesario)
for i in leftovers:
    best = min(range(k), key=lambda c: D[i, medoids[c]])
    clusters[best].append(i)

bin_xy = XY[medoids]

# ────────────────────────────────────────────────────────────────
# 6 · Métricas y salida
# ────────────────────────────────────────────────────────────────
total_dist = sum(D[i, medoids[c]] for c in range(k) for i in clusters[c])
print("\n===================  RESULTADO  ===================")
for c, m in enumerate(medoids):
    print(f"Bin {c + 1}: {len(clusters[c])} árboles ({min_cap}–{max_cap})  en {tuple(bin_xy[c])}")
print(f"Distancia caminada total = {total_dist:.1f} m")

point_cluster = np.empty(N_tot, int)
for c, pts in enumerate(clusters):
    point_cluster[pts] = c

# ────────────────────────────────────────────────────────────────
# 7 · Gráfico
# ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
# filas
for r in range(rows):
    xc = r * dx
    ax.add_patch(patches.Rectangle((xc - w_row / 2, 0), w_row, ROW_TOP, fc="#dddddd", ec="k"))
# árboles por color de cluster
for c in range(k):
    pts = XY[point_cluster == c]
    ax.scatter(pts[:, 0], pts[:, 1], s=9, label=f"cluster {c + 1}")
# bins
ax.scatter(bin_xy[:, 0], bin_xy[:, 1], marker="s", s=130, c="red", ec="k", lw=1.2, label="Bins")
# trayectorias
for i in range(N_tot):
    b = bin_xy[point_cluster[i]]
    p = XY[i]
    d_down = (p[1] - ROW_BOT) + (b[1] - ROW_BOT) + abs(p[0] - b[0])
    d_up = (ROW_TOP - p[1]) + (ROW_TOP - b[1]) + abs(p[0] - b[0])
    path = [
        (p[0], p[1]),
        (p[0], ROW_BOT if d_down <= d_up else ROW_TOP),
        (b[0], ROW_BOT if d_down <= d_up else ROW_TOP),
        (b[0], b[1]),
    ]
    ax.plot(*zip(*path), lw=0.35, alpha=0.25, color="k")

ax.set_aspect("equal")
ax.set_xlim(-dx, (rows - 1) * dx + dx)
ax.set_ylim(-8, ROW_TOP + 5)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title(f"k-medoids restringido  (k={k}, target {PARAMS['N_target']}±{PARAMS['slack']})")
ax.legend(markerscale=1.3, loc="upper right")
plt.tight_layout()
plt.show()
