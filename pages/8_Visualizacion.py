from __future__ import annotations

"""
orchard_app_v24_fixed.py – 🍒 Orchard optimiser (robust grid, no MPSolver import)
================================================================================

Corrección crítica:
-------------------
*Se elimina la referencia a `MPSolver` que no existe en el wrapper de OR‑Tools*
(la constante correcta es `pywraplp.Solver.OPTIMAL`). Ahora el script importa
únicamente `pywraplp` y comprueba el estado del solver con
`status == pywraplp.Solver.OPTIMAL`, evitando el `ImportError`.

Resto de funcionalidades (fallback a equidistant, patrones de entrada,
scenario grid) permanecen idénticas a la versión v24.
"""

import io
import math
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from ortools.linear_solver import pywraplp
except ModuleNotFoundError:
    pywraplp = None

# ────────────────────────────── CONSTANTS
P = dict(
    dt=5,
    horizon=360,
    picker_rate15=10,
    tree_spacing=2,
    bin_cap=300,
    bucket_cap=10,
    walk_speed=60,
)
KG_SLOT = P["picker_rate15"] * P["dt"] / 15

# ────────────────────────────── FIELD GEOMETRY


def tree_coords(rows: int, L: int):
    xs = np.arange(0, L + 1, P["tree_spacing"])
    ys = np.repeat(np.arange(rows), len(xs))
    xs = np.tile(xs, rows)
    return np.column_stack([xs, ys]).astype(float)


def candidate_coords(rows: int, L: int, step: int):
    xs = np.arange(step, L, step)
    ys = np.repeat(np.arange(rows), len(xs))
    xs = np.tile(xs, rows)
    return np.column_stack([xs, ys]).astype(float)


# ────────────────────────────── EQUALLY SPACED BINS (fallback)


def equidistant_bins(rows: int, L: int, k: int):
    per_row = max(1, k // rows)
    xs = np.linspace(P["tree_spacing"], L - P["tree_spacing"], per_row)
    return np.column_stack([np.tile(xs, rows)[:k], np.repeat(np.arange(rows), per_row)[:k]])


# ────────────────────────────── SAFE k‑MEDIAN


def solve_k_median_safe(trees: np.ndarray, cand: np.ndarray, k: int):
    """Devuelve (layout, solved) – `solved` False si no es óptimo."""
    if pywraplp is None or cand.size == 0:
        return np.empty((0, 2)), False
    n, m = trees.shape[0], cand.shape[0]
    dist = np.linalg.norm(trees[:, None] - cand[None, :], axis=2)
    solver = pywraplp.Solver.CreateSolver("SCIP")
    x = {(i, j): solver.BoolVar(f"x_{i}_{j}") for i in range(n) for j in range(m)}
    y = {j: solver.BoolVar(f"y_{j}") for j in range(m)}
    for i in range(n):
        solver.Add(sum(x[i, j] for j in range(m)) == 1)
    for i, j in x:
        solver.Add(x[i, j] <= y[j])
    solver.Add(sum(y[j] for j in range(m)) == k)
    solver.Minimize(sum(dist[i, j] * x[i, j] for i in range(n) for j in range(m)))
    solver.SetTimeLimit(3000)
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        return np.empty((0, 2)), False
    sel = [j for j in range(m) if y[j].solution_value() > 0.5]
    return np.array([cand[j] for j in sel]), True


def robust_layout(rows: int, L: int, k: int, base_step: int):
    """k‑median con retry, luego fallback equidistant."""
    if pywraplp is None:
        st.info("🔧 OR‑Tools no instalado ⇒ usando layout Equidistant")
        return equidistant_bins(rows, L, k)
    step = base_step
    for _ in (1, 2):
        layout, ok = solve_k_median_safe(tree_coords(rows, L), candidate_coords(rows, L, step), k)
        if ok:
            return layout
        step = max(2, step // 2)
    st.warning("⚠️ SCIP no encontró solución ↠ cambiando a Equidistant.")
    return equidistant_bins(rows, L, k)


# ────────────────────────────── PICKER DISTRIBUTIONS


def uniform_assignment(rows: int, pickers: int) -> List[int]:
    base, extra = divmod(pickers, rows)
    return [base + 1 if r < extra else base for r in range(rows)]


def sequential_assignment(rows: int, pickers: int) -> List[int]:
    arr, r = [0] * rows, 0
    while pickers > 0 and r < rows:
        arr[r] += 1
        pickers -= 1
        if arr[r] == 20:
            r += 1
    return arr


# ────────────────────────────── km / picker heuristic


def km_per_picker(rows: int, L: int, bins: np.ndarray, pick_row: List[int]):
    xs = np.arange(0, L + 1, P["tree_spacing"])
    km_total = 0
    ppl = sum(pick_row)
    for r, n in enumerate(pick_row):
        rb = bins[bins[:, 1] == r][:, 0]
        for _ in range(n):
            dist = 0
            for x in xs:
                nearest = L / 2 if rb.size == 0 else np.min(np.abs(rb - x))
                dist += 2 * nearest * P["bucket_cap"] / 20
            km_total += dist / 1000
    return km_total / (ppl or 1)


# minutes


def minutes_to_target(target: int, pickers: int):
    kg_h = P["picker_rate15"] * 4
    return math.ceil(target / (kg_h * pickers) * 60)


# ────────────────────────────── STREAMLIT UI
# vest.set_page_config(page_title='🍒 Orchard optimiser v24‑fixed', layout='wide')
st.title("🍒 Orchard optimiser – robust grid (v24‑fixed)")
with st.sidebar:
    rows = st.slider("Rows", 4, 40, 10)
    L = st.slider("Row length (m)", 100, 400, 200, 10)
    target = st.number_input("Target kg", 1000, 100000, 20000, 1000)
    k_bins = int(math.ceil(target / P["bin_cap"])) + 3
    placement = st.radio("Bins placement", ("Optimised", "Equidistant"))
    density = st.slider("Candidate density (m)", 4, 20, 10, 2)
    pick_min = st.number_input("Pickers min", 1, 500, 20)
    pick_max = st.number_input("Pickers max", 1, 500, 120)
    step = st.number_input("Step", 1, 50, 20)
    run = st.button("▶ Run grid")

if not run:
    st.stop()

bins_xy = (
    robust_layout(rows, L, k_bins, density)
    if placement == "Optimised"
    else equidistant_bins(rows, L, k_bins)
)
if bins_xy.size == 0:
    st.error("❌ No layout generated.")
    st.stop()

# build grid
records = []
for total in range(pick_min, pick_max + 1, step):
    for label, dist_fn in [("Uniform", uniform_assignment), ("Sequential", sequential_assignment)]:
        pick_row = dist_fn(rows, total)
        km = km_per_picker(rows, L, bins_xy, pick_row)
        minutes = minutes_to_target(target, sum(pick_row))
        records.append(dict(strategy=label, pickers=total, minutes=minutes, km_pp=km))

df = pd.DataFrame(records)
best = df.loc[df["km_pp"].idxmin()]

st.subheader("Scenario grid (km/picker)")
st.dataframe(
    df.style.apply(
        lambda r: ["background-color:#d6f5d6" if r.km_pp == best.km_pp else "" for _ in r],
        axis=1,
    ),
)
fig = px.line(
    df,
    x="pickers",
    y="km_pp",
    color="strategy",
    markers=True,
    title="Km/picker vs pickers",
)
fig.add_scatter(
    x=[best.pickers],
    y=[best.km_pp],
    mode="markers",
    marker=dict(size=12, symbol="star", color="gold"),
    name="Best",
)
st.plotly_chart(fig, use_container_width=True)

st.success(
    f"Best → {best.strategy} with {best.pickers} pickers: {best.km_pp:.2f} km/picker • {best.minutes} min",
)

fig_bins = px.scatter(
    pd.DataFrame(bins_xy, columns=["x", "row"]),
    x="x",
    y="row",
    height=450,
    range_x=[0, L],
    range_y=[rows, -1],
)
fig_bins.update_yaxes(title="Row")
st.plotly_chart(fig_bins, use_container_width=True)

with io.BytesIO() as buf:
    df.to_csv(buf, index=False)
    st.download
