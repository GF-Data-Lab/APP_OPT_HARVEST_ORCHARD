#!/usr/bin/env python3
# coloca_bines_balanced.py  (extra ahora opcional)
import argparse
import ast
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


# ----------------------------------------------------------
# 1. Geometría
# ----------------------------------------------------------
def midpoints_rectangles(l, M, alpha, beta, r, a):
    pts = []
    for i, m in enumerate(M):
        xl, xr = alpha + i * (r + a), alpha + i * (r + a) + r
        for j in range(m):
            y = beta + (j + 0.5) * (l / m)
            pts.extend([(xl, y), (xr, y)])
    return np.array(pts)


def rectangle_bounds(l, M, alpha, beta, r, a):
    return [(alpha + i * (r + a), alpha + i * (r + a) + r, beta, beta + l) for i in range(len(M))]


# ----------------------------------------------------------
# 2. Utilidades
# ----------------------------------------------------------
def is_inside_rect(x, y, rect):
    xl, xr, yb, yt = rect
    return xl <= x <= xr and yb <= y <= yt


def push_outside_rects(c, rects, eps):
    x, y = c
    for xl, xr, *_ in rects:
        if xl <= x <= xr:
            x = xl - eps if abs(x - xl) < abs(x - xr) else xr + eps
    return np.array([x, y])


# Balanced K-means
def balanced_kmeans(pts, k, max_iter=100):
    n, _ = pts.shape
    cap = int(np.ceil(n / k))
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(pts)
    centers, labels = km.cluster_centers_, km.labels_
    for _ in range(max_iter):
        counts = np.bincount(labels, minlength=k)
        over = [c for c in range(k) if counts[c] > cap]
        if not over:
            break
        for c in over:
            idxs = np.where(labels == c)[0]
            d = np.linalg.norm(pts[idxs] - centers[c], axis=1)
            for i in idxs[np.argsort(-d)]:
                if counts[c] <= cap:
                    break
                nearest = np.argsort(np.linalg.norm(pts[i] - centers, axis=1))
                for t in nearest:
                    if counts[t] < cap:
                        labels[i] = t
                        counts[c] -= 1
                        counts[t] += 1
                        break
        for c in range(k):
            centers[c] = pts[labels == c].mean(axis=0)
    return labels, centers


# ----------------------------------------------------------
# 3. Pipeline principal
# ----------------------------------------------------------
def place_bins_balanced(l, M, alpha, beta, r, a, k, entries=None, lam=10, eps_factor=0.05):
    if entries is None:
        entries = []
    trees = midpoints_rectangles(l, M, alpha, beta, r, a)
    rects = rectangle_bounds(l, M, alpha, beta, r, a)

    pts = trees.tolist()
    for x, y, n in entries:
        pts.append((x, y))  # sin pesos en versión balanced
    pts = np.array(pts)

    labels, C = balanced_kmeans(pts, k)
    C_adj = np.array([push_outside_rects(c, rects, eps_factor * r) for c in C])
    return pts, labels, C, C_adj, rects


def distances_to_bins(pts, lbl, bins):
    return np.linalg.norm(pts - bins[lbl], axis=1)


# ----------------------------------------------------------
# 4. Visualización
# ----------------------------------------------------------
def draw_rects(ax, rects):
    for xl, xr, yb, yt in rects:
        ax.plot([xl, xr, xr, xl, xl], [yb, yb, yt, yt, yb], "k-")


def plot_scene(pts, lbl, C, C_adj, rects, k):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_aspect("equal")
    draw_rects(ax, rects)
    for c in range(k):
        ax.scatter(*pts[lbl == c].T, s=15, label=f"Cluster {c}")
    ax.scatter(*C.T, marker="x", s=90, c="red", label="Centros")
    ax.scatter(*C_adj.T, marker="D", s=85, edgecolors="k", c="lime", label="Bines")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Bines balanceados")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------
# 5. CLI / dict único
# ----------------------------------------------------------
def parse_args():
    if len(sys.argv) == 1:
        return None
    P = argparse.ArgumentParser(description="Bines balanceados")
    for name in ["l", "alpha", "beta", "r", "a"]:
        P.add_argument(f"--{name}", type=float)
    P.add_argument("--M", type=str)
    P.add_argument("--k", type=int)
    P.add_argument("--extra", type=str, default=None)  # ahora default None
    return P.parse_args()


# ----------------------------------------------------------
# 6. Main
# ----------------------------------------------------------
if __name__ == "__main__":
    cli = parse_args()
    if cli:
        params = {
            "l": cli.l,
            "M": ast.literal_eval(cli.M),
            "alpha": cli.alpha,
            "beta": cli.beta,
            "r": cli.r,
            "a": cli.a,
            "k": cli.k,
            "extra": ast.literal_eval(cli.extra) if cli.extra else [],
        }
    else:
        print('Pega dict con parámetros (puede omitir "extra"):')
        params = ast.literal_eval(input().strip())
        params.setdefault("extra", [])  # si falta, lista vacía

    pts, lbl, C, C_adj, rects = place_bins_balanced(
        params["l"],
        params["M"],
        params["alpha"],
        params["beta"],
        params["r"],
        params["a"],
        params["k"],
        params["extra"],
    )

    d_auto = distances_to_bins(pts, lbl, C_adj)

    plot_scene(pts, lbl, C, C_adj, rects, params["k"])

    print("\nBines (balanceados):")
    for i, c in enumerate(C_adj, 1):
        print(f"  Bin {i}: ({c[0]:.2f}, {c[1]:.2f}) | tamaño cluster = {(lbl == i - 1).sum()} pts")
    print(f"\nMedia global de distancias = {d_auto.mean():.2f} u")

    if input("\n¿Comparar con bines propios? [s/N]: ").strip().lower() == "s":
        print(f"Introduce una lista de {params['k']} coordenadas, ej. [(x1,y1),…]:")
        manual = ast.literal_eval(input().strip())
        if len(manual) != params["k"]:
            print("⚠️  Cantidad incorrecta, fin.")
            sys.exit()
        manual = np.array(manual)
        d_man = distances_to_bins(pts, lbl, manual)
        print("\nMedias por cluster:")
        for c in range(params["k"]):
            print(
                f"Cluster {c + 1}: auto={d_auto[lbl == c].mean():.2f} | manual={d_man[lbl == c].mean():.2f}",
            )
        print(f"\nGlobal auto   = {d_auto.mean():.2f}")
        print(f"Global manual = {d_man.mean():.2f}")
