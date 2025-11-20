"""
Streamlit app that allocates *k* harvest bins (medoids) in an orchard
with blocked (row‑constrained) distances and per‑bin capacity
constraints.

Extended features (2025‑07‑15 → 2025‑07‑15 b)
----------------------------------------------
* **Dynamic orchard blocks by team** – scans `equipo<N>.json` files.
* **Instant preview** – displays a K‑means view of the chosen rows.
* **Row selector** – after picking a block (or manual list) you can
  *elegir las hileras exactas* que se utilizarán antes de ejecutar el
  algoritmo.  A multiselect enumerates each row under the preview; hit
  “Guardar cambios” to refresh the preview.
* Final constrained k‑medoids allocation unchanged.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import plotly.graph_objects as go
# Try to import scikit‑learn for KMeans preview; otherwise use fallback
try:
    from sklearn.cluster import KMeans  # type: ignore
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False

import plotly.graph_objects as go
import numpy as np

def preview_kmeans_plotly(
    XY: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    dx: float,
    row_width: float,
    rows: int,
    dy_tree: float,
) -> go.Figure:
    """Devuelve un Figure Plotly con la vista previa K-means."""
    fig = go.Figure()

    # ── Rectángulos de fondo (“hileras”) ─────────────────────────
    for r in range(rows):
        xc = r * dx
        fig.add_shape(
            type="rect",
            x0=xc - row_width / 2,
            y0=0,
            x1=xc + row_width / 2,
            y1=XY[:, 1].max() + dy_tree if len(XY) else dy_tree,
            line=dict(color="rgba(0,0,0,0.4)", width=1),
            fillcolor="rgba(200,200,200,0.25)",
            layer="below",
        )

    # ── Puntos de árboles por cluster ────────────────────────────
    k = centroids.shape[0]
    for c in range(k):
        pts = XY[labels == c]
        if len(pts):
            fig.add_trace(
                go.Scatter(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    mode="markers",
                    marker=dict(size=4),
                    name=f"Cluster {c+1}",
                )
            )

    # ── Centroides ───────────────────────────────────────────────
    if len(centroids):
        fig.add_trace(
            go.Scatter(
                x=centroids[:, 0],
                y=centroids[:, 1],
                mode="markers",
                marker=dict(symbol="x", size=12, line=dict(width=2)),
                name="Centroides K-means",
            )
        )

    # ── Layout general ───────────────────────────────────────────
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title="Vista previa (K-means)",
        xaxis_title="x (m)",
        yaxis_title="y (m)",
        margin=dict(l=40, r=160, t=60, b=40),
        legend=dict(x=1.02, y=1, xanchor="left"),
        height=500,
    )
    return fig

# ────────────────────────────────────────────────────────────────
# 0 · Utilities to load orchard‑block JSON files by team
# ────────────────────────────────────────────────────────────────

def load_team_files(directory: str | Path = ".") -> Dict[str, List[dict]]:
    """Return mapping ``team_name → list_of_blocks``.

    *Scans for files matching ``equipo*.json``.*
    """
    team_data: Dict[str, List[dict]] = {}
    for path in Path(directory).glob("equipo*.json"):
        try:
            with open(path, "r", encoding="utf‑8") as fh:
                data = json.load(fh)
            blocks = data.get("ORCHARD_BLOCKS", [])
            if isinstance(blocks, list):
                team_data[path.stem] = blocks
        except Exception as exc:  # pragma: no cover
            # Save the error for later feedback
            team_data[path.stem] = exc  # type: ignore[assignment]
    return team_data


def blocks_to_trees_per_face(block: dict) -> List[int]:
    """Convert one block's *hileras* into the ``trees_per_face`` list."""
    pairs = block.get("hileras", [])
    if not pairs:
        return []
    max_row = max(row for row, _ in pairs)
    tpf = [0] * max_row
    for row, count in pairs:
        if 1 <= row <= max_row:
            tpf[row - 1] = count
    return tpf

# ────────────────────────────────────────────────────────────────
# Helper · Compute XY coordinates from trees_per_face
# ────────────────────────────────────────────────────────────────

def orchard_xy(
    trees_per_face: List[int],
    dx_row: float,
    dy_tree: float,
    row_width: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (XY, row_id, side_arr) for plotting / algorithms."""
    rows = len(trees_per_face)
    half_w = row_width / 2.0
    trees: list[tuple[float, float, int, int]] = []
    for r, n in enumerate(trees_per_face):
        xc = r * dx_row
        for side in (0, 1):  # 0‑west, 1‑east
            x = xc + (-half_w if side == 0 else half_w)
            for k in range(int(n)):
                trees.append((x, k * dy_tree, r, side))
    arr = np.asarray(trees, float)
    return arr[:, :2], arr[:, 2].astype(int), arr[:, 3].astype(int)

# ────────────────────────────────────────────────────────────────
# Preview · K‑means clustering (Euclidean) for quick visual check
# ────────────────────────────────────────────────────────────────

def kmeans_preview(XY: np.ndarray, k: int, seed: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Return (labels, centroids) from a simple K‑means run."""
    if _HAS_SKLEARN and len(XY) >= k:
        km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(XY)
        return km.labels_, km.cluster_centers_

    # Fallback lightweight implementation (Lloyd's algorithm)
    rng = np.random.default_rng(seed)
    centroids = XY[rng.choice(len(XY), size=k, replace=False)] if len(XY) else np.empty((0, 2))
    for _ in range(10):
        if not len(XY):
            return np.array([], int), centroids
        d2 = ((XY[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)
        new_centroids = np.vstack([XY[labels == c].mean(axis=0) for c in range(k)])
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return labels, centroids


def plot_preview_kmeans(
    XY: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    dx: float,
    row_width: float,
    rows: int,
    dy_tree: float,
) -> plt.Figure:
    """Return a Matplotlib figure with a K‑means preview."""
    fig, ax = plt.subplots(figsize=(8, 4))

    # Row background blocks
    for r in range(rows):
        xc = r * dx
        ax.add_patch(
            patches.Rectangle(
                (xc - row_width / 2, 0), row_width, XY[:, 1].max() + dy_tree if len(XY) else dy_tree,
                fc="#f5f5f5", ec="none")
        )

    # Scatter trees
    k = centroids.shape[0]
    for c in range(k):
        pts = XY[labels == c]
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=6, label=f"cluster {c+1}")

    # Centroids
    if len(centroids):
        ax.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=130, c="red", ec="k", label="Centroides K‑means")

    ax.set_aspect("equal")
    ax.set_title("Vista previa (K‑means)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    return fig

# ────────────────────────────────────────────────────────────────
# 0 · Utilities to load orchard‑block JSON files by team
# ────────────────────────────────────────────────────────────────

def load_team_files(directory: str | Path = ".") -> Dict[str, List[dict]]:
    """Return mapping ``team_name → list_of_blocks``.

    *Scans for files matching ``equipo*.json``.*
    """
    team_data: Dict[str, List[dict]] = {}
    for path in Path(directory).glob("equipo*.json"):
        try:
            with open(path, "r", encoding="utf‑8") as fh:
                data = json.load(fh)
            blocks = data.get("ORCHARD_BLOCKS", [])
            if isinstance(blocks, list):
                team_data[path.stem] = blocks
        except Exception as exc:  # pragma: no cover
            # Save the error for later feedback
            team_data[path.stem] = exc  # type: ignore[assignment]
    return team_data


def blocks_to_trees_per_face(block: dict) -> List[int]:
    """Convert one block's *hileras* into the ``trees_per_face`` list."""
    pairs = block.get("hileras", [])
    if not pairs:
        return []
    max_row = max(row for row, _ in pairs)
    tpf = [0] * max_row
    for row, count in pairs:
        if 1 <= row <= max_row:
            tpf[row - 1] = count
    return tpf


# ────────────────────────────────────────────────────────────────
# Helper · Compute XY coordinates from trees_per_face
# ────────────────────────────────────────────────────────────────

def orchard_xy(
    trees_per_face: List[int],
    dx_row: float,
    dy_tree: float,
    row_width: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (XY, row_id, side_arr) for plotting / algorithms."""
    rows = len(trees_per_face)
    half_w = row_width / 2.0
    trees: list[tuple[float, float, int, int]] = []
    for r, n in enumerate(trees_per_face):
        xc = r * dx_row
        for side in (0, 1):  # 0‑west, 1‑east
            x = xc + (-half_w if side == 0 else half_w)
            for k in range(int(n)):
                trees.append((x, k * dy_tree, r, side))
    arr = np.asarray(trees, float)
    return arr[:, :2], arr[:, 2].astype(int), arr[:, 3].astype(int)


# ────────────────────────────────────────────────────────────────
# Preview · K‑means clustering (Euclidean) for quick visual check
# ────────────────────────────────────────────────────────────────

def kmeans_preview(XY: np.ndarray, k: int, seed: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Return (labels, centroids) from a simple K‑means run."""
    if _HAS_SKLEARN:
        km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(XY)
        return km.labels_, km.cluster_centers_

    # Fallback lightweight implementation (Lloyd's algorithm)
    rng = np.random.default_rng(seed)
    centroids = XY[rng.choice(len(XY), size=k, replace=False)]
    for _ in range(10):
        # Assign
        d2 = ((XY[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)
        # Update
        new_centroids = np.vstack([XY[labels == c].mean(axis=0) for c in range(k)])
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return labels, centroids


def plot_preview_kmeans(
    XY: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    dx: float,
    row_width: float,
    rows: int,
    dy_tree: float,
) -> plt.Figure:
    """Return a Matplotlib figure with a K‑means preview."""
    fig, ax = plt.subplots(figsize=(8, 4))

    # Row background blocks
    for r in range(rows):
        xc = r * dx
        ax.add_patch(
            patches.Rectangle((xc - row_width / 2, 0), row_width, XY[:, 1].max() + dy_tree, fc="#f5f5f5", ec="none")
        )

    # Scatter trees
    k = centroids.shape[0]
    for c in range(k):
        pts = XY[labels == c]
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=6, label=f"cluster {c+1}")

    # Centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=130, c="red", ec="k", label="Centroids")

    ax.set_aspect("equal")
    ax.set_title("Vista previa (K‑means)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    return fig


# ────────────────────────────────────────────────────────────────
# 1 · Core algorithm utilities (UNCHANGED from previous version)
# ────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────
# 1 · Core algorithm utilities (mostly unchanged from original)
# ────────────────────────────────────────────────────────────────

def run_kmedoids(params):
    """Run constrained k‑medoids and return rich results dict."""

    # ↳ deterministic reproducibility
    np.random.seed(params["seed"])
    random.seed(params["seed"])

    # Orchard geometry -----------------------------------------------------
    rows        = len(params["trees_per_face"])
    dx          = params["dx_row"]
    dy          = params["dy_tree"]
    w_row       = params["row_width"]
    half_w      = w_row / 2.0

    trees = []  # (x, y, row, side)
    for r, n in enumerate(params["trees_per_face"]):
        xc = r * dx
        for side in (0, 1):                               # 0 = west, 1 = east
            x = xc + (-half_w if side == 0 else half_w)
            for k in range(n):
                trees.append((x, k * dy, r, side))
    trees = np.asarray(trees, float)

    XY       = trees[:, :2]
    row_id   = trees[:, 2].astype(int)
    side_arr = trees[:, 3].astype(int)
    N_tot    = len(trees)

    ROW_TOP, ROW_BOT = (max(params["trees_per_face"]) - 1) * dy, 0.0

    # Blocked distance helper ---------------------------------------------
    def blocked_dist(i: int, j: int) -> float:
        p, q = XY[i], XY[j]
        rp, sp, rq, sq = row_id[i], side_arr[i], row_id[j], side_arr[j]
        if rp == rq and sp == sq:
            return np.linalg.norm(p - q)
        d_down = (p[1] - ROW_BOT) + (q[1] - ROW_BOT) + abs(p[0] - q[0])
        d_up   = (ROW_TOP - p[1]) + (ROW_TOP - q[1]) + abs(p[0] - q[0])
        return d_down if d_down < d_up else d_up

    # Pre‑compute distance matrix (symmetric) ------------------------------
    D = np.zeros((N_tot, N_tot))
    for i in range(N_tot):
        for j in range(i + 1, N_tot):
            d = blocked_dist(i, j)
            D[i, j] = D[j, i] = d

    # Capacity checks ------------------------------------------------------
    k          = params["k_bins"]
    min_cap    = params["N_target"] - params["slack"]
    max_cap    = params["N_target"] + params["slack"]
    if k * max_cap < N_tot:
        raise ValueError(f"Capacidad insuficiente: {k}×{max_cap} < {N_tot} árboles. ")

    # k‑medoids heuristic ---------------------------------------------------
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
        clusters, leftovers = [[] for _ in range(k)], []
        # Order by difficulty (gap best–2nd best)
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
        # Enforce minimum by stealing from the biggest cluster
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

    # Assign any leftovers to the nearest bin (may exceed max_cap)
    for i in leftovers:
        best = min(range(k), key=lambda c: D[i, medoids[c]])
        clusters[best].append(i)

    bin_xy = XY[medoids]
    total_dist = sum(D[i, medoids[c]] for c in range(k) for i in clusters[c])

    point_cluster = np.empty(N_tot, int)
    for c, pts in enumerate(clusters):
        point_cluster[pts] = c

    return {
        "clusters": clusters,
        "medoids": medoids,
        "XY": XY,
        "bin_xy": bin_xy,
        "total_dist": total_dist,
        "point_cluster": point_cluster,
        "rows": rows,
        "dx": dx,
        "w_row": w_row,
        "ROW_TOP": ROW_TOP,
        "ROW_BOT": ROW_BOT,
        "k": k,
        "min_cap": min_cap,
        "max_cap": max_cap,
        "N_tot": N_tot,
    }


# ────────────────────────────────────────────────────────────────
# 2 · Plotting helper
# ────────────────────────────────────────────────────────────────

def plot_orchard(
    res: dict,
    params: dict,
    *,
    sep_factor: float = 1.4,
    fig_height_px: int = 600,
    interactive: bool = True,
):
    """Visualise orchard clusters.

    Parameters
    ----------
    res, params
        Outputs from ``run_kmedoids`` and the parameter dict.
    sep_factor
        Multiplier applied to *x* spacing so that rows look farther
        apart.  The underlying coordinates are scaled but the algorithm
        results (distances etc.) remain unchanged.
    fig_height_px
        Pixel height for the *interactive* Plotly figure.
    interactive
        If *True*, return a **Plotly** figure (zoom/pan enabled).  If
        *False*, return a static Matplotlib figure (old behaviour).
    """

    # ────────────────────────────────────────────────────────────────
    # Common geometry
    # ────────────────────────────────────────────────────────────────
    XY, point_cluster, bin_xy = (
        res["XY"].copy(),  # will re‑scale X below
        res["point_cluster"],
        res["bin_xy"].copy(),
    )
    rows, dx, w_row = res["rows"], res["dx"], res["w_row"]
    ROW_TOP, ROW_BOT = res["ROW_TOP"], res["ROW_BOT"]
    k = res["k"]

    # Apply horizontal separation
    XY[:, 0] *= sep_factor
    bin_xy[:, 0] *= sep_factor
    dx *= sep_factor
    w_row *= sep_factor

    # ────────────────────────────────────────────────────────────────
    # Interactive Plotly branch
    # ────────────────────────────────────────────────────────────────
    if interactive:
        fig = go.Figure()

        # Row backgrounds as shapes
        for r in range(rows):
            xc = r * dx
            fig.add_shape(
                type="rect",
                x0=xc - w_row / 2,
                y0=0,
                x1=xc + w_row / 2,
                y1=ROW_TOP,
                line=dict(color="rgba(0,0,0,0.4)", width=1),
                fillcolor="rgba(200,200,200,0.25)",
                layer="below",
            )

        # Scatter clusters
        for c in range(k):
            pts = XY[point_cluster == c]
            if len(pts):
                fig.add_trace(
                    go.Scatter(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        mode="markers",
                        marker=dict(size=4),
                        name=f"Cluster {c+1}",
                    )
                )

        # Scatter bins (medoids)
        fig.add_trace(
            go.Scatter(
                x=bin_xy[:, 0],
                y=bin_xy[:, 1],
                mode="markers",
                marker=dict(symbol="square", size=14, color="red", line=dict(width=1, color="black")),
                name="Bins",
            )
        )

        # Layout tweaks
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(
            title=f"k‑medoids restringido  (k={k}, target {params['N_target']}±{params['slack']})",
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            height=fig_height_px,
            legend=dict(x=1.02, y=1, xanchor="left"),
            margin=dict(l=40, r=160, t=60, b=40),
        )
        return fig

    # ────────────────────────────────────────────────────────────────
    # Static Matplotlib branch (default)
    # ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    # Rows (shaded blocks)
    for r in range(rows):
        xc = r * dx
        ax.add_patch(
            patches.Rectangle(
                (xc - w_row / 2, 0), w_row, ROW_TOP, fc="#dddddd", ec="k"
            )
        )

    # Trees coloured by cluster
    for c in range(k):
        pts = XY[point_cluster == c]
        ax.scatter(pts[:, 0], pts[:, 1], s=9, label=f"cluster {c + 1}")

    # Bins (medoids)
    ax.scatter(
        bin_xy[:, 0],
        bin_xy[:, 1],
        marker="s",
        s=130,
        c="red",
        ec="k",
        lw=1.2,
        label="Bins",
    )

    # Paths (down or up headlands)
    for i in range(len(XY)):
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
    ax.set_title(
        f"k‑medoids restringido  (k={k}, target {params['N_target']}±{params['slack']})"
    )
    ax.legend(
        markerscale=1.3,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    return fig

# ────────────────────────────────────────────────────────────────

# 3 · Streamlit UI

# ────────────────────────────────────────────────────────────────


def main() -> None:  # pragma: no cover
    st.set_page_config(
        page_title="Orchard k‑medoids allocator",
        layout="wide",
        page_icon="🌳",
    )
    st.title("🌳 Orchard k‑medoids bin allocator")

    # -----------------------------------------------------------
    # Sidebar · Equipo / Bloque selection
    # -----------------------------------------------------------
    st.sidebar.header("Fuente de datos")

    team_files = load_team_files()
    manual_option = "Entrada manual"
    team_choices = [manual_option] + sorted([k for k, v in team_files.items() if not isinstance(v, Exception)])
    equipo_sel = st.sidebar.selectbox("Equipo", team_choices)

    trees_per_face_raw: List[int] | None = None
    block_meta: dict[str, Any] | None = None
    block_key: tuple[str, int] | None = None  # (equipo_sel, idx)

    if equipo_sel == manual_option:
        st.sidebar.subheader("Parámetro manual")
        trees_str = st.sidebar.text_input("Trees per face (JSON list)", "[30, 30, 30]")
        try:
            trees_per_face_raw = json.loads(trees_str)
            if not all(isinstance(x, (int, float)) for x in trees_per_face_raw):
                raise ValueError
        except Exception:
            st.sidebar.error("Invalid JSON list. Example: [30, 30, 30]")
            trees_per_face_raw = [30, 30, 30]
    else:
        blocks_or_err = team_files[equipo_sel]
        if isinstance(blocks_or_err, Exception):
            st.sidebar.error(f"Error leyendo {equipo_sel}.json → {blocks_or_err}")
        elif not blocks_or_err:
            st.sidebar.warning(f"{equipo_sel}.json no contiene bloques.")
        else:
            labels = [f"{i+1} · {b['variedad']} (sector {b['sector']})" for i, b in enumerate(blocks_or_err)]
            idx = st.sidebar.selectbox("Bloque", range(len(labels)), format_func=lambda i: labels[i])
            block_meta = blocks_or_err[idx]
            trees_per_face_raw = blocks_to_trees_per_face(block_meta)
            block_key = (equipo_sel, idx)
            with st.sidebar.expander("Detalles bloque", expanded=False):
                st.json({k: v for k, v in block_meta.items() if k != "hileras"})

    # Persist row selection in session_state keyed by (equipo_sel, idx) or manual
    if trees_per_face_raw is not None:
        default_key = (equipo_sel, "manual") if block_key is None else block_key
        # ─── Inicialización de la selección de hileras ───
        if "row_select_key" not in st.session_state or st.session_state.row_select_key != default_key:
            # Primera vez con este bloque → seleccionar todas
            st.session_state.row_select_key = default_key
            st.session_state.selected_rows = list(range(len(trees_per_face_raw)))


        # -------------------------------------------------------
        # Row selector below preview – within an expander & form
        # -------------------------------------------------------
        row_options = [f"Hilera {i+1} ({trees_per_face_raw[i]} árboles)" for i in range(len(trees_per_face_raw))]
        with st.expander("Filtrar hileras", expanded=False):
            with st.form("filter_rows_form"):
                sel_rows = st.multiselect(
                    "Seleccione hileras a incluir",
                    options=list(range(len(row_options))),
                    default=st.session_state.selected_rows,
                    format_func=lambda idx: row_options[idx],
                )
                save_filter = st.form_submit_button("Guardar cambios")
                if save_filter:
                    if sel_rows:
                        st.session_state.selected_rows = sel_rows
                    else:
                        st.warning("Debe seleccionar al menos una hilera.")

    # Build filtered trees_per_face based on selection
    trees_per_face: List[int] | None = None
    if trees_per_face_raw is not None:
        trees_per_face = [
            trees_per_face_raw[i] if i in st.session_state.selected_rows else 0
            for i in range(len(trees_per_face_raw))
        ]

    # -----------------------------------------------------------
    # Sidebar · Remaining numeric parameters
    # -----------------------------------------------------------
    st.sidebar.header("Parámetros del algoritmo")
    dx = st.sidebar.number_input("Row spacing dx (m)", value=4.0, min_value=0.1, step=0.1)
    dy = st.sidebar.number_input("Tree spacing dy (m)", value=2.0, min_value=0.1, step=0.1)
    row_width = st.sidebar.number_input("Row width (m)", value=1.0, min_value=0.1, step=0.1)
    k_bins = st.sidebar.number_input("Number of bins (k)", value=5, min_value=1, step=1)
    N_target = st.sidebar.number_input("Target trees per bin", value=32, min_value=1, step=1)
    slack = st.sidebar.number_input("Slack (± trees)", value=10, min_value=0, step=1)
    seed = st.sidebar.number_input("Random seed", value=1, min_value=0, step=1)
    interactive_plot = st.sidebar.checkbox(
        "Vista interactiva (zoom/pan)", value=False
    )
        # --- quick win ---  Validación inmediata de capacidad
    if trees_per_face and any(trees_per_face):
        max_total = int(k_bins) * (int(N_target) + int(slack))
        total_trees = sum(trees_per_face)
        if total_trees > max_total:
            st.error(
                f"Capacidad insuficiente: {max_total} < {total_trees} árboles.")
    # -----------------------------------------------------------
    # PREVIEW section (K‑means) – shows automatically when data present
    # -----------------------------------------------------------
    if trees_per_face and any(trees_per_face):
        XY, row_id, side_arr = orchard_xy(trees_per_face, dx, dy, row_width)
        if len(XY):
            labels, centroids = kmeans_preview(XY, k=int(k_bins), seed=int(seed))
            preview_fig = plot_preview_kmeans(XY, labels, centroids, dx, row_width, len(trees_per_face), dy)
            st.subheader("Vista previa de hileras seleccionadas (K-means)")
            st.plotly_chart(preview_fig, use_container_width=True)
            st.caption("*Las hileras no seleccionadas se muestran vacías.*")
    elif trees_per_face_raw is not None:
        st.info("No hay hileras seleccionadas.")

    # -----------------------------------------------------------
    # Run button – executes constrained k‑medoids
    # -----------------------------------------------------------
    run_btn = st.sidebar.button("🚀 Ejecutar algoritmo k‑medoids")

    if run_btn:
        if trees_per_face is None or not any(trees_per_face):
            st.error("Debe definir y seleccionar al menos una hilera antes de ejecutar el algoritmo.")
            return

        params = dict(
            trees_per_face=trees_per_face,
            dx_row=dx,
            dy_tree=dy,
            row_width=row_width,
            k_bins=int(k_bins),
            N_target=int(N_target),
            slack=int(slack),
            seed=int(seed),
        )

        with st.spinner("Computando k‑medoids …"):
            try:
                res = run_kmedoids(params)
                import datetime, json, pathlib, uuid

                run_id = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
                run_entry = {
                    "run_id": run_id,
                    "params": params,
                    "metrics": {
                        "total_dist": res["total_dist"],
                        "N_tot": res["N_tot"],
                    },
                }

                # 1) guarda en la sesión
                st.session_state["history"].append(run_entry)

                # 2) botón de descarga inmediata
                json_str = json.dumps(run_entry, indent=2)
                st.download_button(
                    "⬇️ Descargar este run",
                    data=json_str,
                    file_name=f"{run_id}.json",
                )

                # 3) (opcional) guarda en disco local
                SAVE_DIR = pathlib.Path("runs")
                if st.button("💾 Guardar en carpeta runs/"):
                    SAVE_DIR.mkdir(exist_ok=True)
                    (SAVE_DIR / f"{run_id}.json").write_text(json_str)
                    st.toast(f"Run {run_id} guardado en {SAVE_DIR}/")
                # ---------------------------------------------------------------
            except ValueError as e:
                st.error(str(e))
                return

        # -------------------------------------------------------
        # Results summary
        # -------------------------------------------------------
        st.subheader("Resultados finales")
        st.markdown(
            f"**Distancia total caminada:** {res['total_dist']:.1f} m  "
            f"**Total árboles:** {res['N_tot']}")
        cols = st.columns(res["k"])
        for c in range(res["k"]):
            with cols[c]:
                st.metric(
                    label=f"Bin {c + 1}",
                    value=f"{len(res['clusters'][c])} trees",
                    delta=f"({res['min_cap']}–{res['max_cap']})",
                )

        # -------------------------------------------------------
        # Plot – final allocation
        # -------------------------------------------------------
        fig = plot_orchard(res, params, interactive=interactive_plot)
        
        if interactive_plot:
            st.plotly_chart(fig, use_container_width=True)   # Plotly (zoom, pan)
        else:
            st.pyplot(fig)                                   # Matplotlib clásico
        import pandas as pd
        import plotly.graph_objects as go
        
        if st.session_state["history"]:
            st.subheader("Historial de ejecuciones (sesión actual)")
        
            # ▶️ Tabla interactiva
            df = pd.json_normalize(st.session_state["history"])
            st.dataframe(
                df[["run_id", "params.k_bins", "metrics.total_dist"]],
                use_container_width=True,
            )
        
            # ▶️ Gráfico comparativo
            fig_hist = go.Figure()
            fig_hist.add_trace(
                go.Scatter(
                    x=df["params.k_bins"],
                    y=df["metrics.total_dist"],
                    mode="markers+lines",
                    text=df["run_id"],
                    name="Runs",
                )
            )
            fig_hist.update_layout(
                xaxis_title="k_bins",
                yaxis_title="Distancia total (m)",
                title="Distancia vs. k_bins",
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
# --- memoria de sesión para runs ---------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []        # lista de dicts por ejecución
# -----------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    main()
