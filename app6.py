# app.py  ·  versión integrada 2025‑06
import math, numpy as np, pandas as pd, matplotlib.pyplot as plt, streamlit as st
import io, xlsxwriter
# ╔══════════════════════ 0 · BARRA LATERAL ═══════════════════════╗
st.title("Simulador de cosecha · comparación normal_4 vs otros layouts")

# --- Huerto ------------------------------------------------------------
st.sidebar.header("Huerto")
arb_txt = st.sidebar.text_input("Árboles por cara (ej. 40,45)", "40,45")
try:
    ARBOLES_CARA = [int(x) for x in arb_txt.split(",") if x.strip()]
except ValueError:
    st.sidebar.error("Sólo enteros separados por coma"); st.stop()

A_X = st.sidebar.number_input("A_X · dist. hileras (m)", .5, 10.0, 4.0, .1)
D_Y = st.sidebar.number_input("D_Y · dist. árbol‑árbol (m)", .5, 10.0, 2.0, .1)

# --- Rendimiento individual -------------------------------------------
st.sidebar.header("Rendimiento individual")
SPEED_MEAN = st.sidebar.number_input("Vel. media (m/min)", 1.0, 500.0, 4000/60, 1.)
SPEED_SD   = st.sidebar.number_input("SD velocidad", 0.0, 200.0, 500/60, 1.)
TREES_MEAN = st.sidebar.number_input("Árboles/tote media", .1, 10.0, 1.5, .1)
TREES_SD   = st.sidebar.number_input("SD árboles/tote", 0.0, 10.0, .5, .1)

# --- Kg ---------------------------------------------------------------
st.sidebar.header("Parámetros de kg")
KG_TREE = st.sidebar.number_input("Kg por árbol", 1.0, 200.0, 25.0, .5)
KG_TOTE = st.sidebar.number_input("Kg por tote", 1.0, 500.0, 18.0, .5)

# --- Jornada ----------------------------------------------------------
st.sidebar.header("Jornada")
START_MIN = st.sidebar.number_input("Inicio (min)", 0, 24*60, 8*60, 1)
END_MIN   = st.sidebar.number_input("Fin (min)",    0, 24*60,13*60, 1)

# --- Bines ------------------------------------------------------------
st.sidebar.header("Bines")
BIN_CAP        = st.sidebar.number_input("Cap. bin (totes)", 1, 400, 20, 1)
BIN_TIME_LIMIT = st.sidebar.number_input("Lím. tiempo bin (min)", 1, 240, 30, 1)
BINS_DISP      = st.sidebar.number_input("Bines disponibles (layouts fijos)", 1, 400, 20, 1)

# --- Layouts & mano de obra ------------------------------------------
st.sidebar.header("Comparación de layouts")
LAYOUTS_ALL = ["normal_4", "calle_central", "bajo_hilera", "extremos_calle"]

LAYOUTS_ON = st.sidebar.multiselect(
    "Elige qué layouts fijos comparar (normal_4 siempre incluido)",
    options=[l for l in LAYOUTS_ALL if l != "normal_4"],
    default=[l for l in LAYOUTS_ALL if l != "normal_4"]
)

USE_KMEDOIDS = st.sidebar.checkbox(
    "Añadir layout «k‑medoids» (bines óptimos, sin cruzar hileras)",
    value=True
)

pp_txt = st.sidebar.text_input("Personas por cara (ej. 2,3,6)", "2,3,6")
try:
    PP_LIST = sorted({int(x) for x in pp_txt.split(",") if x.strip()})
except ValueError:
    st.sidebar.error("Lista de personas inválida"); st.stop()

PRICE_TOTE = st.sidebar.number_input("Pago por tote ($)", 0.0, 50.0, 1.25, .05)

# --- Detalle ----------------------------------------------------------
st.sidebar.header("Detalle backlog / kg")
DETAIL_LAYOUTS = ["normal_4"] + LAYOUTS_ON + (["kmedoids"] if USE_KMEDOIDS else [])
LAY_DET = st.sidebar.selectbox("Layout detallado", DETAIL_LAYOUTS)
PP_DET  = st.sidebar.selectbox("PP‑cara detallado", PP_LIST)

# --- Otros ------------------------------------------------------------
SEED     = st.sidebar.number_input("Seed", 0, 999_999, 11, 1)
INVERT_X = st.sidebar.checkbox("Invertir eje X", True)

# ╔══════════════════════ 1 · GEOMETRÍA BÁSICA ════════════════════════╗
def make_bin_sets(arboles_cara, a_x=4., d_y=2., gap_x=2., gap_y=2., y_off=-5.):
    n=len(arboles_cara); xc=np.arange(n)*a_x; cx=(n-1)*a_x/2
    L=(max(arboles_cara)-1)*d_y
    return {
        "normal_4":      [(cx-gap_x/2,y_off+gap_y),(cx-gap_x/2,y_off),
                          (cx+gap_x/2,y_off),     (cx+gap_x/2,y_off+gap_y)],
        "calle_central": [(cx, L/3),(cx, 2*L/3)],
        "bajo_hilera":   [(x,y_off) for x in xc],
        "extremos_calle":[(x,0) for x in xc]+[(x,L) for x in xc],
    }
BIN_SETS = make_bin_sets(ARBOLES_CARA, A_X, D_Y)

def trees_xy():
    pts=[]
    for h,n in enumerate(ARBOLES_CARA):
        for s in (-.5,.5):
            for k in range(n):
                pts.append((h*A_X+s, k*D_Y))
    return np.array(pts,float)
TREES = trees_xy()

# ╔══════════════════════ 2 · MÉTRICA Y K‑MEDOIDS ═════════════════════╗
def row_path_dist(p1, p2, L):
    x1,y1=p1; x2,y2=p2
    return min(abs(y1-0)+abs(x1-x2)+abs(y2-0),
               abs(y1-L)+abs(x1-x2)+abs(y2-L))

def kmedoids_rows(points, k, rng, iters=30):
    medoids = points[rng.choice(len(points), k, replace=False)].copy()
    L = points[:,1].max()
    for _ in range(iters):
        labels = np.array([np.argmin([row_path_dist(p,m,L) for m in medoids]) for p in points])
        new = medoids.copy()
        for j in range(k):
            clust = points[labels==j]
            if clust.size:
                D = np.array([[row_path_dist(a,b,L) for b in clust] for a in clust])
                new[j] = clust[np.argmin(D.sum(1))]
        if np.allclose(new, medoids): break
        medoids = new
    return medoids

# ╔══════════════════════ 3 · SIMULADOR ═══════════════════════════════╗
def fatigue(now, shift, rng):
    HARV_START,HARV_END,HARV_SD=20,25,5
    mu = HARV_START + np.clip((now-START_MIN)/shift,0,1)*(HARV_END-HARV_START)
    return max(1, rng.normal(mu, HARV_SD))

def run_sim(ppface:int, bins_xy:np.ndarray, rng)->dict:
    W = ppface*len(ARBOLES_CARA)*2
    n_bins=len(bins_xy)
    bin_load=np.zeros(n_bins,int); bin_start=[None]*n_bins; open_bin=0
    face_to_idx={}
    for i,(h,s) in enumerate([(h,s) for h in range(len(ARBOLES_CARA)) for s in (0,1)
                              for _ in range(ARBOLES_CARA[h])]):
        face_to_idx.setdefault((h,s),[]).append(i)
    for lst in face_to_idx.values(): lst.sort(key=lambda ix: TREES[ix][1])
    workers_face=[fc for fc in face_to_idx for _ in range(ppface)]
    speed=rng.normal(SPEED_MEAN,SPEED_SD,W).clip(1,None)
    ptr={}; pos=[]
    for fc in workers_face:
        i=ptr.get(fc,0); ptr[fc]=i+1
        pos.append(tuple(TREES[face_to_idx[fc][i]]))
    w_time=np.zeros(W); picked=np.zeros(len(TREES),bool); remain=len(TREES)
    totes_w=np.zeros(W,int); km_w=np.zeros(W)
    bin_events=[[(0,"in_pos",0)] for _ in range(n_bins)]
    shift=END_MIN-START_MIN
    trees_per_tote=lambda:1 if rng.normal(TREES_MEAN,TREES_SD)<1.5 else 2

    while remain and np.isfinite(w_time).any():
        w=int(np.nanargmin(w_time)); now=w_time[w]
        fc=workers_face[w]; lst=face_to_idx[fc]; sel=[]
        while lst and len(sel)<trees_per_tote():
            ix=lst.pop(0)
            if not picked[ix]: sel.append(ix)
        if not sel: w_time[w]=np.inf; continue
        for ix in sel: picked[ix]=True
        remain-=len(sel)
        p0=np.array(pos[w]); dist=np.linalg.norm(p0-TREES[sel[0]])
        if len(sel)==2: dist+=np.linalg.norm(TREES[sel[1]]-TREES[sel[0]])
        last=TREES[sel[-1]]
        b=open_bin
        if bin_start[b] and now-bin_start[b]>=BIN_TIME_LIMIT: b+=1
        if b<n_bins and bin_load[b]>=BIN_CAP: b+=1
        if b<n_bins and b!=open_bin: open_bin=b
        if open_bin>=n_bins:
            w_time[w]=now+5; pos[w]=tuple(last); continue
        dist+=np.linalg.norm(bins_xy[open_bin]-last)
        if bin_start[open_bin] is None: bin_start[open_bin]=now
        walk=dist/speed[w]; harv=fatigue(now,shift,rng); fin=now+walk+harv
        w_time[w]=fin; pos[w]=tuple(bins_xy[open_bin])
        totes_w[w]+=1; km_w[w]+=dist/1000
        bin_load[open_bin]+=1
        bin_events[open_bin].append((fin,"load",bin_load[open_bin]))
        if bin_load[open_bin]==BIN_CAP:
            bin_events[open_bin].append((fin,"full",BIN_CAP))
    horas=(START_MIN+w_time[np.isfinite(w_time)].max()-START_MIN)/60
    pick_df=pd.DataFrame({"Picker":np.arange(W),"Totes":totes_w,"Km":km_w,
                          "Ingreso $":totes_w*PRICE_TOTE})
    bins_df=pd.DataFrame({"Bin":np.arange(n_bins),"Totes totales":bin_load})
    return dict(horas=horas, pickers=pick_df, bins=bins_df,
                bin_events=bin_events, km_total=km_w.sum(),
                totes_total=totes_w.sum())


# ╔══════════════════════ 4 · BINES POR LAYOUT ════════════════════════╗
TOTAL_KG      = len(TREES)*KG_TREE
TOTES_THEO    = math.ceil(TOTAL_KG / KG_TOTE)
BINS_MIN_THEO = math.ceil(TOTES_THEO / BIN_CAP)
rng_bins      = np.random.default_rng(SEED+123)

def bins_for(layout_name:str):
    if layout_name == "kmedoids":
        return kmedoids_rows(TREES, BINS_MIN_THEO, rng_bins)
    else:
        return np.array(BIN_SETS[layout_name], float)

# lista definitiva y ordenada: normal_4 base, luego otros, luego kmedoids
LAYOUTS_RUN = ["normal_4"] + [l for l in LAYOUTS_ON if l!="normal_4"]
if USE_KMEDOIDS: LAYOUTS_RUN.append("kmedoids")

# ╔══════════════════════ 5 · SIMULACIONES GLOBALES ════════════════════╗
res={lay:pd.DataFrame(index=PP_LIST,
         columns=["Trabajadores","Horas","Totes","Km","Costo $"])
      for lay in LAYOUTS_RUN}

st.write(f"**Kg totales:** {TOTAL_KG:,.0f} kg · "
         f"Totes teóricos: {TOTES_THEO:,d} · "
         f"Bines teóricos: {BINS_MIN_THEO}")

prog=st.progress(0.0, text="Simulando...")
for i,pp in enumerate(PP_LIST,1):
    for lay in LAYOUTS_RUN:
        rng=np.random.default_rng(SEED)
        sim=run_sim(pp, bins_for(lay), rng)
        res[lay].loc[pp]= (pp*len(ARBOLES_CARA)*2,
                           sim["horas"], sim["totes_total"],
                           sim["km_total"],
                           sim["pickers"]["Ingreso $"].sum())
    prog.progress(i/len(PP_LIST))
prog.empty()

# ╔══════════════════════ 6 · TABLA Y GRÁFICOS GLOBALES ════════════════╗
st.subheader("Comparativa global (normal_4 es el escenario base)")
tabT,tabH,tabC = st.tabs(["Tabla","Horas","Costo $"])

with tabT:
    st.dataframe(pd.concat(res,axis=1).style.format("{:.2f}"))

with tabH:
    fig,ax=plt.subplots(figsize=(7,4))
    for lay,df in res.items():
        ax.plot(df.index,df["Horas"],marker='o',label=lay)
    if INVERT_X: ax.invert_xaxis()
    ax.set_xlabel("Personas/cara"); ax.set_ylabel("Horas")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); st.pyplot(fig)

with tabC:
    fig,ax=plt.subplots(figsize=(7,4))
    for lay,df in res.items():
        ax.plot(df.index,df["Costo $"],marker='s',label=lay)
    if INVERT_X: ax.invert_xaxis()
    ax.set_xlabel("Personas/cara"); ax.set_ylabel("Costo total ($)")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); st.pyplot(fig)

# ╔══════════════════════ 7 · DETALLE BACKLOG / KG ═════════════════════╗
st.subheader(f"Detalle – layout **{LAY_DET}**, {PP_DET} personas/cara")

sim_det = run_sim(PP_DET, bins_for(LAY_DET),
                  np.random.default_rng(SEED))

col1,col2 = st.columns(2)

with col1:  # mapa
    bx = bins_for(LAY_DET)
    fig,ax=plt.subplots(figsize=(5,8))
    ax.scatter(TREES[:,0],TREES[:,1],s=8,alpha=.25,label="Árboles")
    ax.scatter(bx[:,0],bx[:,1],marker='s',s=120,color='tab:red',label="Bines")
    for i,(x,y) in enumerate(bx): ax.text(x,y,f"{i}",ha='center',va='center',
                                          color='white',fontsize=8)
    ax.set_aspect('equal'); ax.set_title("Mapa huerto"); ax.grid(alpha=.2); ax.legend()
    st.pyplot(fig)

with col2:  # tabla bines
    st.dataframe(sim_det["bins"].style.format("{:.0f}"))
    used  = (sim_det["bins"]["Totes totales"]>0).sum()
    st.success(f"Bines usados: {used}/{len(bx)}   ·   "
               f"Kg cosechados: {sim_det['totes_total']*KG_TOTE:,.0f}")

# ── Gráfico A: kg por bin ─────────────────────────────────────────────
st.markdown("#### A) Kg acumulados en cada bin")
figA, axA = plt.subplots(figsize=(9,4))
for b,ev in enumerate(sim_det["bin_events"]):
    t,k=[0],[0]
    for ti,typ,val in ev:
        if typ in ("load","full"):
            t.append(ti); k.append(val*KG_TOTE)
    axA.step(t,k,where='post',label=f"Bin {b}")
axA.set_xlabel("Minutos"); axA.set_ylabel("Kg"); axA.grid(alpha=.3)
axA.legend(ncol=6,fontsize=7); figA.tight_layout(); st.pyplot(figA)

# ── Gráfico B: kg restantes ───────────────────────────────────────────
st.markdown("#### B) Kg restantes en el huerto")
figB, axB = plt.subplots(figsize=(9,3))
kg_rest = TOTAL_KG; tr=[0]; kr=[kg_rest]
for ti,_,_ in sorted([e for sub in sim_det["bin_events"] for e in sub
                      if e[1] in ("load","full")], key=lambda x:x[0]):
    kg_rest -= KG_TOTE; tr.append(ti); kr.append(kg_rest)
axB.step(tr,kr,where='post',color='black')
axB.set_xlabel("Minutos"); axB.set_ylabel("Kg restantes"); axB.grid(alpha=.3)
figB.tight_layout(); st.pyplot(figB)

# ── Pickers ───────────────────────────────────────────────────────────
st.markdown("#### Pickers – km & ingreso")
st.dataframe(sim_det["pickers"].style.format({"Km":"{:.2f}",
                                              "Ingreso $":"{:.2f}"}))
figP,axP=plt.subplots(figsize=(8,4))
axP.bar(sim_det["pickers"]["Picker"], sim_det["pickers"]["Km"])
axP.set_xlabel("Picker"); axP.set_ylabel("Km"); axP.set_title("Km por picker")
figP.tight_layout(); st.pyplot(figP)
# ═════════════  Exportar a Excel  ═════════════


if st.button("Descargar resultados en Excel"):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # Hoja 1: resumen global
        pd.concat(res, axis=1).to_excel(writer, sheet_name="Resumen_global")
        # Hoja 2: bins del layout detallado
        sim_det["bins"].to_excel(writer, sheet_name=f"Bins_{LAY_DET}")
        # Hoja 3: pickers del layout detallado
        sim_det["pickers"].to_excel(writer, sheet_name=f"Pickers_{LAY_DET}")
    output.seek(0)
    st.download_button(
        label="📥 Descargar Excel",
        data=output,
        file_name="simulacion_cosecha.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
