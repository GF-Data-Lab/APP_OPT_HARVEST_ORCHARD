# app.py  ·  versión 2025‑06
import streamlit as st, numpy as np, pandas as pd, matplotlib.pyplot as plt, math

# ═════════════════ 0 · INPUT UI ═════════════════
st.title("Simulador integral de cosecha · layouts, bines y kg")

# --- Huerto ---
st.sidebar.header("Huerto")
arboles_txt = st.sidebar.text_input("Árboles por cara (ej. 40,45)", "40,45")
try:
    ARBOLES_CARA = [int(x) for x in arboles_txt.split(",") if x.strip()]
except ValueError:
    st.sidebar.error("Usa enteros separados por coma"); st.stop()

A_X = st.sidebar.number_input("A_X – dist. hileras (m)", .5, 10.0, 4.0, .1)
D_Y = st.sidebar.number_input("D_Y – dist. árbol‑árbol (m)", .5, 10.0, 2.0, .1)

# --- Rendimientos base ---
st.sidebar.header("Rendimiento individual")
SPEED_MEAN = st.sidebar.number_input("Velocidad media (m/min)", 1.0, 500.0, 4000/60, 1.)
SPEED_SD   = st.sidebar.number_input("SD velocidad", 0.0, 200.0, 500/60, 1.)
TREES_MEAN = st.sidebar.number_input("Árboles/tote media", .1, 10.0, 1.5, .1)
TREES_SD   = st.sidebar.number_input("SD árboles/tote", 0.0, 10.0, .5, .1)

# --- KG ---
st.sidebar.header("Parámetros de kg")
KG_TREE = st.sidebar.number_input("Kg por árbol", 1.0, 200.0, 25.0, 0.5)
KG_TOTE = st.sidebar.number_input("Kg por tote", 1.0, 500.0, 18.0, 0.5)

# --- Jornada ---
st.sidebar.header("Jornada")
START_MIN = st.sidebar.number_input("Inicio jornada (min)", 0, 24*60, 8*60, 1)
END_MIN   = st.sidebar.number_input("Fin jornada (min)",    0, 24*60,13*60, 1)

# --- Bines ---
st.sidebar.header("Bines")
BIN_CAP        = st.sidebar.number_input("Cap. bin (totes)", 1, 400, 20, 1)
BIN_TIME_LIMIT = st.sidebar.number_input("Lím. tiempo bin (min)", 1, 240, 30, 1)
BINS_USER      = st.sidebar.number_input("Bines disponibles", 1, 400, 20, 1)
#BINS_KM = math.ceil(TOTAL_TOTES_EST / BIN_CAP)   # mismo cálculo teórico

# --- Layouts & workforce ---
st.sidebar.header("Layouts y mano de obra")
LAYOUTS_ALL = ["normal_4", "calle_central", "bajo_hilera", "extremos_calle"]
LAYOUTS_ON  = st.sidebar.multiselect("Layouts a comparar", LAYOUTS_ALL, default=LAYOUTS_ALL)

pp_text = st.sidebar.text_input("Personas por cara a simular (ej. 2,3,6,20)", "2,3,6")
try:
    PP_LIST = sorted({int(x) for x in pp_text.split(",") if x.strip()})
    if not PP_LIST: raise ValueError
except ValueError:
    st.sidebar.error("Introduce enteros separados por coma"); st.stop()

PRICE_TOTE = st.sidebar.number_input("Pago por tote ($)", 0.0, 50.0, 1.25, .05)

# --- Detalle (1 layout + 1 pp) ---
st.sidebar.header("Detalle backlog / kg")
LAY_DET = st.sidebar.selectbox("Layout detallado", LAYOUTS_ON)
PP_DET  = st.sidebar.selectbox("PP‑cara detallado", PP_LIST)

SEED = st.sidebar.number_input("Seed", 0, 999_999, 11, 1)
INVERT_X = st.sidebar.checkbox("Invertir eje X", True)

# ═════════════════ 1 · GEOMETRÍA ═════════════════
def make_bin_sets(arboles_cara, a_x=4., d_y=2., gap_x=2., gap_y=2., y_off=-5.):
    n=len(arboles_cara); xc=np.arange(n)*a_x; cx=(n-1)*a_x/2; L=(max(arboles_cara)-1)*d_y
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
# ── distancia 'row‑path'   (sin atravesar hileras) ───────────────────
def row_path_dist(p1, p2, L):
    """
    Distancia mínima entre p1=(x1,y1) y p2=(x2,y2)
    suponiendo que sólo puedo desplazarme:
       · en Y dentro de la misma hilera
       · en X únicamente por cualquiera de los extremos y=0 o y=L
    """
    x1,y1 = p1; x2,y2 = p2
    # camino vía extremo inferior
    d0 = abs(y1-0) + abs(x1-x2) + abs(y2-0)
    # camino vía extremo superior
    dL = abs(y1-L) + abs(x1-x2) + abs(y2-L)
    return min(d0, dL)

# ── k‑medoids simplificado con la métrica anterior ────────────────────
def kmedoids_rows(points, k, rng, iters=30):
    """
    points : np.ndarray (N,2)  -> coordenadas de los árboles
    k      : nº de bines
    Devuelve array (k,2) con las posiciones elegidas (medoides).
    """
    # inicializamos con k puntos aleatorios
    idx = rng.choice(len(points), k, replace=False)
    medoids = points[idx].copy()
    L = points[:,1].max()               # longitud de las hileras

    for _ in range(iters):
        # Paso 1: asignación a medoid más cercano
        labels = np.empty(len(points), int)
        for i,p in enumerate(points):
            d = [row_path_dist(p, m, L) for m in medoids]
            labels[i] = int(np.argmin(d))

        # Paso 2: para cada cluster buscamos el punto que minimiza distancias
        new_medoids = medoids.copy()
        for j in range(k):
            cluster_pts = points[labels == j]
            if len(cluster_pts) == 0:
                continue
            # matriz de distancias dentro del cluster
            D = np.array([[row_path_dist(a,b,L) for b in cluster_pts] for a in cluster_pts])
            new_medoids[j] = cluster_pts[np.argmin(D.sum(axis=1))]
        if np.allclose(new_medoids, medoids):
            break
        medoids = new_medoids
    return medoids
USE_KMEDOIDS = st.sidebar.checkbox(
    "Ubicar bines con k‑medoids (prohibido cruzar hileras)",
    value=False)
# ═════════════════ 2 · SIMULADOR ═════════════════
def fatigue(now, shift, rng):
    HARV_START, HARV_END, HARV_SD = 20,25,5
    mu = HARV_START + np.clip((now-START_MIN)/shift, 0,1)*(HARV_END-HARV_START)
    return max(1, rng.normal(mu, HARV_SD))

def run_sim(ppface:int, bins_xy:np.ndarray, rng)->dict:
    W = ppface*len(ARBOLES_CARA)*2
    n_bins=len(bins_xy); bin_load=np.zeros(n_bins,int); bin_start=[None]*n_bins; open_bin=0
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
    totes_worker=np.zeros(W,int); km_worker=np.zeros(W)
    bin_events=[[(0,"in_pos",0) ] for _ in range(n_bins)]
    shift=END_MIN-START_MIN

    def trees_per_tote(): return 1 if rng.normal(TREES_MEAN,TREES_SD)<1.5 else 2

    while remain and np.isfinite(w_time).any():
        w=int(np.nanargmin(w_time)); now=w_time[w]; fc=workers_face[w]; lst=face_to_idx[fc]
        sel=[]
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
        if open_bin>=n_bins: w_time[w]=now+5; pos[w]=tuple(last); continue
        dist+=np.linalg.norm(bins_xy[open_bin]-last)
        if bin_start[open_bin] is None: bin_start[open_bin]=now
        walk=dist/speed[w]; harv=fatigue(now,shift,rng); fin=now+walk+harv
        w_time[w]=fin; pos[w]=tuple(bins_xy[open_bin])
        km_worker[w]+=dist/1000; totes_worker[w]+=1
        bin_load[open_bin]+=1
        bin_events[open_bin].append((fin,"load",bin_load[open_bin]))
        if bin_load[open_bin]==BIN_CAP:
            bin_events[open_bin].append((fin,"full",BIN_CAP))
    horas=(START_MIN+w_time[np.isfinite(w_time)].max()-START_MIN)/60
    pick_df=pd.DataFrame({"Picker":np.arange(W),
                          "Totes":totes_worker,
                          "Km":km_worker,
                          "Ingreso $":totes_worker*PRICE_TOTE})
    bins_df=pd.DataFrame({"Bin":np.arange(n_bins),
                          "Totes totales":bin_load})
    return dict(horas=horas,
                pickers=pick_df,
                bins=bins_df,
                bin_events=bin_events,
                km_total=km_worker.sum(),
                totes_total=totes_worker.sum())

# ═════════════════ 3 · SIMULACIONES MULTI‑PP ═════════════════
rng0=np.random.default_rng(SEED)
res={lay:pd.DataFrame(index=PP_LIST,
                      columns=["Trabajadores","Horas","Totes","Km","Costo $"])
      for lay in LAYOUTS_ON}

TOTAL_KG = len(TREES)*KG_TREE
TOTAL_TOTES_EST = math.ceil(TOTAL_KG / KG_TOTE)
BINS_NEEDED_THEO = math.ceil(TOTAL_TOTES_EST / BIN_CAP)
BINS_KM = math.ceil(TOTAL_TOTES_EST / BIN_CAP)   # mismo cálculo teórico

st.write(f"**Kg totales del huerto:** {TOTAL_KG:,.0f} kg  "
         f"≙ {TOTAL_TOTES_EST:,d} totes  "
         f"→ se necesitan ~{BINS_NEEDED_THEO} bines "
         f"(cap. {BIN_CAP} totes c/u)")

prog=st.progress(0.)
for i,pp in enumerate(PP_LIST,1):
    for lay in LAYOUTS_ON:
        rng=np.random.default_rng(SEED)
        bins_xy=np.array(BIN_SETS[lay],float)
        if len(bins_xy)<BINS_NEEDED_THEO:
            extra=BINS_NEEDED_THEO-len(bins_xy)
            # añadimos "fantasmas" fuera del campo para que el modelo tenga bins
            bins_xy=np.vstack([bins_xy,
                               np.column_stack((np.full(extra,-10.),
                                                np.linspace(0, max(TREES[:,1]), extra)))])
        sim=run_sim(pp,bins_xy,rng)
        cost=sim["pickers"]["Ingreso $"].sum()
        res[lay].loc[pp]= (pp*len(ARBOLES_CARA)*2, sim["horas"],
                           sim["totes_total"], sim["km_total"], cost)
    prog.progress(i/len(PP_LIST))
prog.empty()


if USE_KMEDOIDS:
    rng_bins = np.random.default_rng(SEED+123)
    bins_custom = kmedoids_rows(TREES, BINS_KM, rng_bins)
    BIN_SETS["kmedoids"] = bins_custom
    if "kmedoids" not in LAYOUTS_ON:
        LAYOUTS_ON.append("kmedoids")

# ═════════════════ 4 · TABLA Y GRÁFICAS GLOBAL ═════════════════
st.subheader("Comparativa global")
tabT,tabH,tabC=st.tabs(["Tabla","Horas","Costo $"])
with tabT: st.dataframe(pd.concat(res,axis=1).style.format("{:.2f}"))
with tabH:
    fig,ax=plt.subplots(figsize=(7,4))
    for lay,df in res.items():
        ax.plot(df.index,df["Horas"],marker='o',label=lay)
    if INVERT_X: ax.invert_xaxis()
    ax.set_xlabel("Personas por cara"); ax.set_ylabel("Horas"); ax.legend(); ax.grid(alpha=.3); st.pyplot(fig)
with tabC:
    fig,ax=plt.subplots(figsize=(7,4))
    for lay,df in res.items():
        ax.plot(df.index,df["Costo $"],marker='s',label=lay)
    if INVERT_X: ax.invert_xaxis()
    ax.set_xlabel("Personas por cara"); ax.set_ylabel("Costo total ($)"); ax.legend(); ax.grid(alpha=.3); st.pyplot(fig)

# ═════════════════ 5 · DETALLE KG / BACKLOG ═════════════════
st.subheader(f"Detalle – layout **{LAY_DET}**, {PP_DET} personas/cara")

rng_det=np.random.default_rng(SEED)
bins_det=np.array(BIN_SETS[LAY_DET],float)
if len(bins_det)<BINS_NEEDED_THEO:
    extra=BINS_NEEDED_THEO-len(bins_det)
    bins_det=np.vstack([bins_det,
                        np.column_stack((np.full(extra,-10.),np.linspace(0,max(TREES[:,1]),extra)))])
sim=run_sim(PP_DET,bins_det,rng_det)

colM,colKg=st.columns(2)
with colM:
    fig,ax=plt.subplots(figsize=(5,8))
    ax.scatter(TREES[:,0],TREES[:,1],s=8,alpha=.25,label="Árboles")
    ax.scatter(bins_det[:,0],bins_det[:,1],marker='s',s=120,color='tab:red',label="Bines")
    for i,(x,y) in enumerate(bins_det): ax.text(x,y,f"{i}",ha='center',va='center',color='white',fontsize=8)
    ax.set_aspect('equal'); ax.grid(alpha=.2); ax.legend()
    ax.set_title("Mapa del huerto"); st.pyplot(fig)

with colKg:
    st.dataframe(sim["bins"].style.format("{:.0f}"))
    st.write(f"**Bines utilizados:** { (sim['bins']['Totes totales']>0).sum() } / {len(bins_det)}")
    st.write(f"**Totes totales cosechados:** {sim['totes_total']:,d}  "
             f"≙ {sim['totes_total']*KG_TOTE:,.0f} kg")

# Timeline KG
st.markdown("### Timeline de kg por bin & kg restantes")
fig,ax=plt.subplots(figsize=(9,4))
kg_rest=TOTAL_KG
t_prev=0
for b,ev in enumerate(sim["bin_events"]):
    t,kg=[0],[0]
    for ti,typ,val in ev:
        if typ in ("load","full"):
            t.append(ti); kg.append(val*KG_TOTE)
    ax.step(t,kg,where='post',label=f"Bin {b}")
# kg restantes
times_sorted=sorted({t for ev in sim["bin_events"] for t,typ,_ in ev})
kg_remaining=[]
for t in times_sorted:
    totes_hasta=sum(val for ev in sim["bin_events"] for ti,typ,val in ev
                    if typ in("load","full") and ti<=t)
    kg_remaining.append(TOTAL_KG - totes_hasta*KG_TOTE)
ax.step(times_sorted,kg_remaining,where='post',color='black',linewidth=2,label="Kg restantes (huerto)")
ax.set_xlabel("Minutos desde inicio"); ax.set_ylabel("Kg"); ax.grid(alpha=.3); ax.legend(ncol=3,fontsize=8)
fig.tight_layout(); st.pyplot(fig)

# Pickers
st.markdown("### Pickers – km & ingreso")
st.dataframe(sim["pickers"].style.format({"Km":"{:.2f}","Ingreso $":"{:.2f}"}))
fig,ax=plt.subplots(figsize=(7,4))
ax.bar(sim["pickers"]["Picker"],sim["pickers"]["Km"])
ax.set_xlabel("Picker"); ax.set_ylabel("Km"); ax.set_title("Km recorridos por picker"); st.pyplot(fig)
