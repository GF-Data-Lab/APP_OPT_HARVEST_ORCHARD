#!/usr/bin/env python3
# harvestflow_seqbins.py
#
# 6 hileras · 24 trabajadores · 4 bines (20 totes c/u)
# Bines se llenan secuencialmente y se “cierran” a los 20 totes
# o 30 minutos después del primer tote depositado.

import numpy as np, matplotlib.pyplot as plt, matplotlib.patches as patches

# ---------------- 1. Parámetros ----------------
n_hileras, arboles_cara = 6, [10, 15, 20, 10, 10, 8]
a, d               = 4.0, 2.0
workers, ppl_face  = 24, 2
speed_mpm          = 4000/60        # m/min (4 km/h)
t_mu_am, t_mu_pm   = 20, 30
t_sd, noon_split   = 5, 150
start_day          = 8*60           # 08:00 en minutos
BIN_CAP            = 20
BIN_TIME_LIMIT     = 30             # min
np.random.seed(11)

# ---------------- 2. Geometría -----------------
trees, face_map = [], []
for i in range(n_hileras):
    xc=i*a
    for s in (0,1):
        x=xc+(-0.5 if s==0 else 0.5)
        for k in range(arboles_cara[i]):
            trees.append((x,k*d)); face_map.append((i,s))
trees=np.array(trees)

# Bines (2×2)
width,gap=(n_hileras-1)*a,2.0; cx=width/2
bins=np.array([[cx-gap/2,-5.25+gap],[cx-gap/2,-5.25],
               [cx+gap/2,-5.25],[cx+gap/2,-5.25+gap]])
n_bins=len(bins)
bin_load=np.zeros(n_bins,int)
bin_open_idx=0           # bin actualmente abierto
bin_start_time=[None]*n_bins

# ---------------- 3. Índices por cara ----------
face_to_idx={}
for idx,fc in enumerate(face_map):
    face_to_idx.setdefault(fc,[]).append(idx)
for lst in face_to_idx.values():
    lst.sort(key=lambda ix: trees[ix][1])      # Y asc.

# ---------------- 4. Trabajadores --------------
faces=[(i,s) for i in range(n_hileras) for s in (0,1)]
workers_faces=[fc for fc in faces for _ in range(ppl_face)]
worker_pos=[]; face_counter={}
for fc in workers_faces:
    lst=face_to_idx[fc]; k=face_counter.get(fc,0)
    worker_pos.append(tuple(trees[lst[k % len(lst)]]))
    face_counter[fc]=k+1

# ---------------- 5. Estado --------------------
worker_time=np.zeros(workers)
picked=np.zeros(len(trees),bool)
logs=[[] for _ in range(workers)]
lost_totes=0

def trees_per_tote(): return 1 if np.random.rand()<0.5 else 2
def harvest(now):      return max(1,np.random.normal(t_mu_am if now<noon_split else t_mu_pm, t_sd))
def hhmm(m):           h,m=divmod(int(m),60); return f"{h:02d}:{m:02d}"

remaining=len(trees)

# ---------------- 6. Simulación ----------------
while remaining and np.isfinite(worker_time).any():
    w=int(np.nanargmin(worker_time)); now=worker_time[w]; fc=workers_faces[w]
    lst=face_to_idx[fc]

    # ------ seleccionar 1-2 árboles no cosechados ------
    sel=[]
    while lst and len(sel)<trees_per_tote():
        ix=lst.pop(0)
        if not picked[ix]:
            sel.append(ix)
    if not sel: worker_time[w]=np.inf; continue
    assert len(sel)<=2
    for ix in sel: picked[ix]=True
    remaining-=len(sel)

    # ------ distancia ------
    p0=np.array(worker_pos[w]); dist=np.linalg.norm(p0-trees[sel[0]])
    if len(sel)==2: dist+=np.linalg.norm(trees[sel[1]]-trees[sel[0]])
    last=trees[sel[-1]]

    # ------ bin secuencial ------
    b_idx=bin_open_idx
    # ¿Se cierra por tiempo?
    if bin_start_time[b_idx] is not None and now-bin_start_time[b_idx]>=BIN_TIME_LIMIT:
        b_idx+=1
    # ¿Se cierra por capacidad?
    if b_idx<n_bins and bin_load[b_idx]>=BIN_CAP:
        b_idx+=1
    # Abrir el nuevo bin si existe
    if b_idx<n_bins and b_idx!=bin_open_idx:
        bin_open_idx=b_idx
    # Si no quedan bines con espacio -> tote se pierde
    if bin_open_idx>=n_bins:
        lost_totes+=1
        worker_time[w]=now+5  # pierde 5 min buscando
        worker_pos[w]=tuple(last)
        continue

    # mover al bin actual
    dist+=np.linalg.norm(bins[bin_open_idx]-last)
    if bin_start_time[bin_open_idx] is None:
        bin_start_time[bin_open_idx]=now  # primer tote llega

    walk=dist/speed_mpm; harv=harvest(now); tot=walk+harv; fin=now+tot
    logs[w].append({"trees":len(sel),"walk":dist,"walk_min":walk,
                    "harv":harv,"total":tot,"fin":hhmm(start_day+fin),
                    "bin":bin_open_idx+1})
    bin_load[bin_open_idx]+=1
    worker_time[w]=fin; worker_pos[w]=tuple(bins[bin_open_idx])

# ---------------- 7. Resultados ----------------
global_fin=worker_time[np.isfinite(worker_time)].max()
print(f"\n⏱  Tiempo global: {global_fin/60:.2f} h (fin {hhmm(start_day+global_fin)})")
print(f"📦 Bines llenados secuencialmente (cap 20 totes): {bin_load.tolist()}")
if lost_totes: print(f"⚠️  Totes perdidos por falta de bines: {lost_totes}")

for i,(fc,lg) in enumerate(zip(workers_faces,logs),1):
    if not lg: print(f"👷 Trabajador {i:02d} Cara {fc} — sin actividad\n"); continue
    d_tot=sum(l['walk'] for l in lg)
    print(f"\n👷 Trabajador {i:02d} Cara {fc} — Totes {len(lg):2d} · Dist {d_tot:.1f} m")
    for j,l in enumerate(lg,1):
        print(f"  Tote {j:02d}: {l['trees']} árbol(es) · {l['walk']:.1f} m "
              f"({l['walk_min']:.1f} min) · cosecha {l['harv']:.1f} min · "
              f"total {l['total']:.1f} min · fin {l['fin']} · Bin {l['bin']}")

# ---------------- 8. Gráfico final -------------
fig,ax=plt.subplots(figsize=(10,6))
ax.scatter(*trees.T,s=10,c='gray',label='Árboles')
max_y=max(arboles_cara)*d
for i in range(n_hileras):
    ax.add_patch(patches.Rectangle((i*a-1,0),2,max_y,edgecolor='black',facecolor='none'))
for i,(bx,by) in enumerate(bins,1):
    ax.scatter(bx,by,marker='s',s=130,c='red')
    ax.text(bx,by+0.3,f'Bin {i}\n{bin_load[i-1]}/20',ha='center')
wp=np.array(worker_pos)
ax.scatter(wp[:,0],wp[:,1],c='blue',s=60,label='Pos. final')
ax.set_aspect('equal'); ax.set_xlim(-2,width+2); ax.set_ylim(-6,max_y+2)
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
ax.set_title('Mapa final – relleno secuencial de bines')
ax.legend(); plt.tight_layout(); plt.show()
