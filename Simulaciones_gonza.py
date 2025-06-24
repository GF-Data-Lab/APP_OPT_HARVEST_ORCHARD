#!/usr/bin/env python3
import numpy as np, matplotlib.pyplot as plt
import matplotlib.patches as patches

# ───────── 0 · constantes del huerto ─────────
ARBOLES_CARA = [40, 45]   # ← cambia aquí
A_X, D_Y     = 4.0, 2.0
SPEED_MEAN, SPEED_SD = 4000/60, 500/60
TREES_MEAN, TREES_SD = 1.5, 0.5
HARV_START, HARV_END, HARV_SD = 20, 25, 5
START_MIN, END_MIN  = 8*60, 13*60        # 08:00 – 13:00
SHIFT_LEN           = END_MIN - START_MIN
BIN_CAP, BIN_TIME_LIMIT = 20, 30
np.random.seed(11)

# ───────── 1 · generador de layouts ─────────
def make_bin_sets(arboles_cara, a_x=4.0, d_y=2.0,
                  gap_x=2.0, gap_y=2.0, y_off=-5.0):
    n  = len(arboles_cara)
    xc = np.arange(n)*a_x
    cx = (n-1)*a_x/2
    L  = (max(arboles_cara)-1)*d_y
    return {
        "normal_4": [(cx-gap_x/2,y_off+gap_y),(cx-gap_x/2,y_off),
                     (cx+gap_x/2,y_off),(cx+gap_x/2,y_off+gap_y)],
        "calle_central": [(cx, L/3),(cx, 2*L/3)],
        "bajo_hilera":   [(x,y_off) for x in xc],
        "extremos_calle":[(x,0) for x in xc]+[(x, L) for x in xc],
    }

BIN_SETS = make_bin_sets(ARBOLES_CARA, A_X, D_Y)
LAYOUTS  = list(BIN_SETS.keys())

# ───────── 2 · simulador (devuelve tiempo global en horas) ─────────
def run_sim(ppface:int, layout:str)->float:
    workers      = ppface * len(ARBOLES_CARA)*2
    bins         = np.array(BIN_SETS[layout], float)
    n_bins       = len(bins)
    bin_load     = np.zeros(n_bins,int)
    bin_start    = [None]*n_bins
    open_bin     = 0

    # árboles y caras
    trees, face_map = [],[]
    for h in range(len(ARBOLES_CARA)):
        xc=h*A_X
        for s in (0,1):
            x=xc+(-0.5 if s==0 else 0.5)
            for k in range(ARBOLES_CARA[h]):
                trees.append((x,k*D_Y)); face_map.append((h,s))
    trees=np.array(trees)
    face_to_idx={}
    for idx,fc in enumerate(face_map):
        face_to_idx.setdefault(fc,[]).append(idx)
    for lst in face_to_idx.values(): lst.sort(key=lambda ix: trees[ix][1])

    # trabajadores
    faces_all=[(i,s) for i in range(len(ARBOLES_CARA)) for s in (0,1)]
    workers_faces=[fc for fc in faces_all for _ in range(ppface)]
    speed=np.random.normal(SPEED_MEAN,SPEED_SD,workers).clip(1,None)
    worker_pos=[]; ptr={}
    for fc in workers_faces:
        lst=face_to_idx[fc]; c=ptr.get(fc,0)
        worker_pos.append(tuple(trees[lst[c]])); ptr[fc]=c+1

    # estado
    w_time=np.zeros(workers); picked=np.zeros(len(trees),bool)
    remain=len(trees)
    def trees_per_tote():
        return 1 if np.random.normal(TREES_MEAN,TREES_SD)<1.5 else 2
    def fatigue(t):
        frac=np.clip((t-START_MIN)/SHIFT_LEN,0,1)
        mu = HARV_START+frac*(HARV_END-HARV_START)
        return max(1,np.random.normal(mu,HARV_SD))

    while remain and np.isfinite(w_time).any():
        w=int(np.nanargmin(w_time)); now=w_time[w]
        fc=workers_faces[w]; lst=face_to_idx[fc]
        sel=[]
        while lst and len(sel)<trees_per_tote():
            ix=lst.pop(0)
            if not picked[ix]: sel.append(ix)
        if not sel: w_time[w]=np.inf; continue
        for ix in sel: picked[ix]=True
        remain-=len(sel)
        p0=np.array(worker_pos[w])
        dist=np.linalg.norm(p0-trees[sel[0]])
        if len(sel)==2: dist+=np.linalg.norm(trees[sel[1]]-trees[sel[0]])
        last=trees[sel[-1]]

        # bines secuenciales
        b=open_bin
        if bin_start[b] and now-bin_start[b]>=BIN_TIME_LIMIT: b+=1
        if b<n_bins and bin_load[b]>=BIN_CAP: b+=1
        if b<n_bins and b!=open_bin: open_bin=b
        if open_bin>=n_bins: w_time[w]=now+5; worker_pos[w]=tuple(last); continue
        dist+=np.linalg.norm(bins[open_bin]-last)
        if bin_start[open_bin] is None: bin_start[open_bin]=now
        walk=dist/speed[w]; harv=fatigue(now); tot=walk+harv
        fin=now+tot
        w_time[w]=fin; worker_pos[w]=tuple(bins[open_bin])
        bin_load[open_bin]+=1

    global_fin=START_MIN+w_time[np.isfinite(w_time)].max()
    return (global_fin-START_MIN)/60   # h

# ───────── 3 · barrido de PPFACE y gráfico ─────────
MAX_PPFACE = 6
pp_values  = np.arange(1, MAX_PPFACE+1)
times = {lay: [run_sim(pp, lay) for pp in pp_values] for lay in LAYOUTS}

# impresión de tabla resumen
print("Tiempo global (h) según personas/cara:")
for lay in LAYOUTS:
    print(f"{lay:15}: " + "  ".join(f"{t:.2f}" for t in times[lay]))

# gráfico
plt.figure(figsize=(7,4))
for lay in LAYOUTS:
    plt.plot(pp_values, times[lay], marker='o', label=lay)
plt.gca().invert_xaxis()   # opcional: menos personas a la derecha
plt.grid(alpha=.3); plt.legend()
plt.xlabel("Personas por cara"); plt.ylabel("Horas para terminar")
plt.title("Productividad vs nº de trabajadores por cara")
plt.tight_layout(); plt.show()
